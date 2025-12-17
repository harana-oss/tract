#![allow(unsafe_op_in_unsafe_fn)]

use super::smallvec::SmallVec;
use crate::ml::bias;
use crate::ml::matmul::MatmulTiled;
use crate::ml::{math, softmax};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::hash::{Hash, Hasher};
use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransformLR {
    None,
    Softmax,
    Logistic,
}

pub fn parse_post_transform_lr(s: &str) -> TractResult<PostTransformLR> {
    match s {
        "NONE" => Ok(PostTransformLR::None),
        "SOFTMAX" => Ok(PostTransformLR::Softmax),
        "LOGISTIC" => Ok(PostTransformLR::Logistic),
        _ => bail!("Invalid post_transform: {}", s),
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegressorData {
    pub coefficients: Arc<Tensor>, // original weights (targets, features) or (features) when targets==1
    pub intercepts: Option<Arc<Tensor>>, // (targets) or (1)
    pub targets: usize,
    pub feat_c: usize,
    pub coefficients_raw: Arc<[f32]>, // row-major (targets, feat_c)
    pub coefficients_t: Arc<[f32]>,   // transposed (feat_c, targets)
    pub prefer_transposed: bool,
    pub packed_tiled: Option<MatmulTiled>,
    pub bias_scalar: f32,
    pub intercepts_vec: Option<Arc<[f32]>>, // length == targets
}

impl Hash for LinearRegressorData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash only canonical sources; derived buffers excluded.
        self.coefficients.hash(state);
        self.intercepts.hash(state);
        self.targets.hash(state);
        self.feat_c.hash(state);
    }
}

impl LinearRegressorData {
    pub fn new(
        coefficients: Arc<Tensor>,
        intercepts: Option<Arc<Tensor>>,
        targets: usize,
    ) -> TractResult<Self> {
        let coeff_slice = coefficients.as_slice::<f32>()?;
        let rank = coefficients.rank();
        let (t, feat_c) = if rank == 2 {
            let shape = coefficients.shape();
            (shape[0], shape[1])
        } else if rank == 1 {
            (targets, coeff_slice.len() / targets.max(1))
        } else {
            bail!("Unsupported coefficients rank {}", rank)
        };
        if t != targets {
            bail!("Coefficient target mismatch: expected {} got {}", targets, t);
        }
        if t * feat_c != coeff_slice.len() {
            bail!(
                "Inconsistent coefficients size t={} feat_c={} len={}",
                t,
                feat_c,
                coeff_slice.len()
            );
        }
        let mut transposed = vec![0f32; coeff_slice.len()];
        for cls in 0..t {
            for f in 0..feat_c {
                transposed[f * t + cls] = coeff_slice[cls * feat_c + f];
            }
        }
        let coefficients_raw: Arc<[f32]> = coeff_slice.to_vec().into();
        let coefficients_t: Arc<[f32]> = transposed.into();

        // Heuristics consistent with LinearClassifier
        let prefer_transposed = t >= 4 && feat_c >= 16;
        let packed_tiled = if t >= 8 && feat_c >= 16 {
            Some(MatmulTiled::new_from_transposed(&coefficients_t, feat_c, t))
        } else {
            None
        };

        // Precompute bias representation
        let mut bias_scalar = 0.0f32;
        let mut intercepts_vec: Option<Arc<[f32]>> = None;
        if let Some(b) = &intercepts {
            let bsl = b.as_slice::<f32>()?;
            if bsl.len() == 1 {
                bias_scalar = bsl[0];
            } else if bsl.len() == t {
                intercepts_vec = Some(bsl.to_vec().into());
            } else {
                bail!("Unsupported intercepts length {}, expected 1 or {}", bsl.len(), t);
            }
        }
        Ok(Self {
            coefficients: coefficients.clone(), // clone to avoid move while slice borrowed
            intercepts,
            targets,
            feat_c,
            coefficients_raw,
            coefficients_t,
            prefer_transposed,
            packed_tiled,
            bias_scalar,
            intercepts_vec,
        })
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearRegressorOp {
    pub data: LinearRegressorData,
    pub post_transform: PostTransformLR,
}

impl LinearRegressorOp {
    // Thread-local row buffer to handle non-contiguous inputs without heap churn
    #[inline(always)]
    fn with_rowbuf<F: FnOnce(&mut [f32]) -> R, R>(len: usize, f: F) -> R {
        thread_local! {
            static ROWBUF_TL_LR: std::cell::RefCell<SmallVec<f32, 256>> = std::cell::RefCell::new(SmallVec::new());
        }
        ROWBUF_TL_LR.with(|rb| {
            let mut v = rb.borrow_mut();
            let mut h = v.handle();
            if h.len() < len {
                h.resize(len, 0.0);
            }
            f(&mut h[..len])
        })
    }

    // Bias helpers moved to crate::ml::bias

    pub fn eval(&self, input: ArrayViewD<f32>) -> TractResult<Tensor> {
        // Normalize input view to 2D (n, c)
        let input_2d: ArrayView2<f32> = if input.ndim() == 1 {
            let len = input.len();
            input.into_shape_with_order((1, len))?
        } else {
            input.into_dimensionality()?
        };
        let n = input_2d.shape()[0];
        let c = input_2d.shape()[1];
        if c != self.data.feat_c {
            bail!("Feature mismatch: model {} != input {}", self.data.feat_c, c);
        }
        let t = self.data.targets;

        // Allocate output without zeroing
        let mut out_tensor = unsafe { Tensor::uninitialized::<f32>(&[n, t]) }?;
        let out_slice = out_tensor.as_slice_mut::<f32>()?;

        // Matmul
        unsafe {
            let coef_rm: &[f32] = &self.data.coefficients_raw;
            let coef_t: &[f32] = &self.data.coefficients_t;
            let packed = self.data.packed_tiled.as_ref();
            let use_t = self.data.prefer_transposed;

            if let Some(x) = input_2d.as_slice() {
                if t == 1 && packed.is_none() {
                    // Dot per row
                    let w = &coef_rm[..c];
                    for i in 0..n {
                        let row = &x[i * c..(i + 1) * c];
                        out_slice[i * t] = math::dot_neon(row, w);
                    }
                } else if let Some(pk) = packed {
                    pk.gemm_rows_contig(x, n, out_slice);
                } else if use_t {
                    math::matmul_rows_neon_contig_t(x, n, c, coef_t, t, out_slice);
                } else {
                    math::matmul_rows_neon_contig(x, n, c, coef_rm, t, out_slice);
                }
            } else {
                // Gathered path for strided/non-contiguous ndarray views
                let base = input_2d.as_ptr();
                // ndarray strides are in element counts already
                let nd_strides = input_2d.strides();
                let s0 = if n == 1 { 0 } else { nd_strides[0] as isize };
                let s1 = nd_strides[1] as isize;
                Self::with_rowbuf(c, |rowbuf| {
                    if t == 1 && packed.is_none() {
                        let w = &coef_rm[..c];
                        for i in 0..n {
                            // gather row i into rowbuf
                            let mut k = 0usize;
                            while k < c {
                                let off = i as isize * s0 + k as isize * s1;
                                rowbuf[k] = *base.offset(off);
                                k += 1;
                            }
                            out_slice[i] = math::dot_neon(&rowbuf[..c], w);
                        }
                    } else if let Some(pk) = packed {
                        pk.gemm_rows_gather(base, n, s0, s1, out_slice, rowbuf);
                    } else if use_t {
                        math::matmul_rows_neon_gather_t(
                            base, n, c, s0, s1, coef_t, t, out_slice, rowbuf,
                        );
                    } else {
                        math::matmul_rows_neon_gather(
                            base, n, c, s0, s1, coef_rm, t, out_slice, rowbuf,
                        );
                    }
                });
            }

            // Bias
            for row in out_slice.chunks_mut(t) {
                if let Some(b) = &self.data.intercepts_vec {
                    bias::add_bias_neon_row(row, b);
                } else if self.data.bias_scalar != 0.0 {
                    bias::add_scalar_bias_neon(row, self.data.bias_scalar);
                }
            }

            // Post-transform
            match self.post_transform {
                PostTransformLR::None => {}
                PostTransformLR::Logistic => {
                    softmax::logistic_inplace_rows(out_slice, n, t);
                }
                PostTransformLR::Softmax => {
                    softmax::softmax_inplace_rows(out_slice, n, t);
                }
            }
        }

        Ok(out_tensor)
    }
}

impl Op for LinearRegressorOp {
    fn name(&self) -> StaticName {
        "LinearRegressor".into()
    }
    op_as_typed_op!();
}

impl EvalOp for LinearRegressorOp {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let input = input.to_array_view::<f32>()?;
        let y = self.eval(input)?;
        Ok(tvec!(y.into_tvalue()))
    }
}

impl TypedOp for LinearRegressorOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        Ok(tvec!(f32::fact(&[n.clone(), self.data.targets.into()])))
    }
    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("coefficients"),
        TypeName::Scalar.tensor().named("intercepts"),
        TypeName::Integer.named("targets"),
        TypeName::String.named("post_transform"),
    ]
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LinearRegressorOp,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let coefficients =
        ast.konst_variable(format!("{}_coefficients", node.name), &op.data.coefficients)?;
    let mut args = vec![input, coefficients];
    let named = vec![
        ("targets", numeric(op.data.targets as i64)),
        (
            "post_transform",
            string(match op.post_transform {
                PostTransformLR::None => "NONE",
                PostTransformLR::Softmax => "SOFTMAX",
                PostTransformLR::Logistic => "LOGISTIC",
            }),
        ),
    ];
    if let Some(intercepts) = &op.data.intercepts {
        let intercepts_var = ast.konst_variable(format!("{}_intercepts", node.name), intercepts)?;
        args.push(intercepts_var);
    }
    Ok(Some(invocation("tract_onnx_ml_linear_regressor", &args, &named)))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let coefficients: Arc<Tensor> = invocation.named_arg_as(builder, "coefficients")?;
    let intercepts = invocation.named_arg_as(builder, "intercepts").ok();
    let targets: i64 = invocation.named_arg_as(builder, "targets")?;
    let post_transform: String = invocation.named_arg_as(builder, "post_transform")?;
    let post_transform = parse_post_transform_lr(&post_transform)?;
    let targets_usize = usize::try_from(targets).context("targets out of range")?;
    let data = LinearRegressorData::new(coefficients.clone(), intercepts.clone(), targets_usize)?;
    let op = LinearRegressorOp { data, post_transform };
    builder.wire(op, &[input])
}

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_linear_regressor",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}
