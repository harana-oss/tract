use crate::ml::softmax;
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
        Ok(Self {
            coefficients: coefficients.clone(), // clone to avoid move while slice borrowed
            intercepts,
            targets,
            feat_c,
            coefficients_raw,
            coefficients_t,
        })
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearRegressorOp {
    pub data: LinearRegressorData,
    pub post_transform: PostTransformLR,
}

impl LinearRegressorOp {
    fn eval_internal(&self, input: ArrayViewD<f32>) -> TractResult<Array2<f32>> {
        // Avoid use-after-move by separating len fetch.
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
        let w = &self.data.coefficients_raw;
        let mut out = Array2::<f32>::zeros((n, t));
        // simple loops (targets usually small)
        for i in 0..n {
            let row = input_2d.row(i);
            for target in 0..t {
                let mut acc = 0f32;
                for f in 0..c {
                    acc += row[f] * w[target * c + f];
                }
                out[[i, target]] = acc;
            }
        }
        if let Some(intercepts) = &self.data.intercepts {
            let b = intercepts.as_slice::<f32>()?;
            if b.len() == t {
                for i in 0..n {
                    for target in 0..t {
                        out[[i, target]] += b[target];
                    }
                }
            } else if b.len() == 1 {
                let bias = b[0];
                for v in out.iter_mut() {
                    *v += bias;
                }
            }
        }
        match self.post_transform {
            PostTransformLR::None => {}
            PostTransformLR::Logistic => {
                let slice = out.as_slice_mut().unwrap();
                softmax::logistic_inplace_rows(slice, n, t);
            }
            PostTransformLR::Softmax => {
                let slice = out.as_slice_mut().unwrap();
                softmax::softmax_inplace_rows(slice, n, t);
            }
        }
        Ok(out)
    }

    pub fn eval(&self, input: ArrayViewD<f32>) -> TractResult<Tensor> {
        Ok(Tensor::from(self.eval_internal(input)?.into_dyn()))
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
