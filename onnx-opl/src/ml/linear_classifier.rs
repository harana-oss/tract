#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: linear_classifier requires target_arch = aarch64");

use super::smallvec::SmallVec;
use crate::ml::math;
use crate::ml::matmul::MatmulTiled;
use crate::ml::{argmax, softmax};
use std::arch::aarch64::*;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use tract_nnef::internal::*;

thread_local! {
    static ROWBUF_TL: RefCell<SmallVec<f32, 256>> = RefCell::new(SmallVec::new());
    static COMPACT_TL: RefCell<SmallVec<f32, 256>> = RefCell::new(SmallVec::new());
}

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_linear_classifier",
        &parameters(),
        &[("label", TypeName::Scalar.tensor()), ("scores", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

pub fn parse_post_transform(s: &str) -> TractResult<PostTransformLC> {
    match s {
        "NONE" => Ok(PostTransformLC::None),
        "SOFTMAX" => Ok(PostTransformLC::Softmax),
        "LOGISTIC" => Ok(PostTransformLC::Logistic),
        "SOFTMAX_ZERO" | "PROBIT" => bail!("SOFTMAX_ZERO and PROBIT unsupported"),
        _ => bail!("Invalid post_transform: {}", s),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransformLC {
    None,
    Softmax,
    Logistic,
}

#[derive(Debug, Clone)]
pub struct LinearClassifierData {
    pub labels: Arc<Tensor>,
    pub coefficients: Arc<Tensor>,
    pub intercepts: Option<Arc<Tensor>>,
    pub eff_e: usize,
    pub feat_c: usize,
    pub coefficients_raw: Arc<[f32]>,
    pub coefficients_t: Arc<[f32]>,
    pub labels_i64: Option<Arc<[i64]>>,
    pub labels_str: Option<Arc<[String]>>,
    pub packed_tiled: Option<MatmulTiled>,
    pub prefer_transposed: bool,
    pub m_out: usize,
    pub binary_compact: bool,
    pub bias_scalar: f32,
    pub intercepts_eff_e: Option<Arc<[f32]>>,
    pub binary_intercepts_2: Option<[f32; 2]>,
}

impl Hash for LinearClassifierData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.labels.hash(state);
        self.coefficients.hash(state);
        self.intercepts.hash(state);
        self.eff_e.hash(state);
        self.feat_c.hash(state);
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearClassifier {
    pub data: LinearClassifierData,
    pub multi_class: i64,
    pub post_transform: PostTransformLC,
}

impl LinearClassifier {
    #[inline(always)]
    fn tensor1_from_vec_string(v: Vec<String>) -> TractResult<Tensor> {
        let len = v.len();
        let mut t = unsafe { Tensor::uninitialized::<String>(&[len]) }?;
        t.as_slice_mut::<String>()?.clone_from_slice(&v);
        Ok(t)
    }

    #[inline(always)]
    fn tensor1_from_vec_i64(v: Vec<i64>) -> TractResult<Tensor> {
        let len = v.len();
        let mut t = unsafe { Tensor::uninitialized::<i64>(&[len]) }?;
        t.as_slice_mut::<i64>()?.clone_from_slice(&v);
        Ok(t)
    }

    pub fn n_classes(&self) -> usize {
        self.data.labels.len()
    }

    pub fn labels(&self) -> &Arc<Tensor> {
        &self.data.labels
    }

    #[inline(always)]
    fn with_rowbuf<F: FnOnce(&mut [f32]) -> R, R>(len: usize, f: F) -> R {
        ROWBUF_TL.with(|rb| {
            let mut v = rb.borrow_mut();
            let mut h = v.handle();
            if h.len() < len {
                h.resize(len, 0.0);
            }
            f(&mut h[..len])
        })
    }

    #[inline(always)]
    fn with_compact<F: FnOnce(&mut [f32]) -> R, R>(len: usize, f: F) -> R {
        COMPACT_TL.with(|cb| {
            let mut v = cb.borrow_mut();
            let mut h = v.handle();
            if h.len() < len {
                h.resize(len, 0.0);
            }
            f(&mut h[..len])
        })
    }

    /// Vectorized bias addition for multiple values
    #[inline(always)]
    unsafe fn add_bias_neon(scores: &mut [f32], bias: &[f32]) {
        let n = scores.len();
        debug_assert_eq!(n, bias.len());
        let mut i = 0;

        while i + 16 <= n {
            let s0 = vld1q_f32(scores.as_ptr().add(i));
            let s1 = vld1q_f32(scores.as_ptr().add(i + 4));
            let s2 = vld1q_f32(scores.as_ptr().add(i + 8));
            let s3 = vld1q_f32(scores.as_ptr().add(i + 12));
            let b0 = vld1q_f32(bias.as_ptr().add(i));
            let b1 = vld1q_f32(bias.as_ptr().add(i + 4));
            let b2 = vld1q_f32(bias.as_ptr().add(i + 8));
            let b3 = vld1q_f32(bias.as_ptr().add(i + 12));
            let r0 = vaddq_f32(s0, b0);
            let r1 = vaddq_f32(s1, b1);
            let r2 = vaddq_f32(s2, b2);
            let r3 = vaddq_f32(s3, b3);
            vst1q_f32(scores.as_mut_ptr().add(i), r0);
            vst1q_f32(scores.as_mut_ptr().add(i + 4), r1);
            vst1q_f32(scores.as_mut_ptr().add(i + 8), r2);
            vst1q_f32(scores.as_mut_ptr().add(i + 12), r3);
            i += 16;
        }

        while i + 4 <= n {
            let s = vld1q_f32(scores.as_ptr().add(i));
            let b = vld1q_f32(bias.as_ptr().add(i));
            let r = vaddq_f32(s, b);
            vst1q_f32(scores.as_mut_ptr().add(i), r);
            i += 4;
        }

        while i < n {
            scores[i] += bias[i];
            i += 1;
        }
    }

    /// Vectorized scalar bias addition
    #[inline(always)]
    unsafe fn add_scalar_bias_neon(scores: &mut [f32], bias: f32) {
        let n = scores.len();
        let bias_vec = vdupq_n_f32(bias);
        let mut i = 0;

        while i + 16 <= n {
            let s0 = vld1q_f32(scores.as_ptr().add(i));
            let s1 = vld1q_f32(scores.as_ptr().add(i + 4));
            let s2 = vld1q_f32(scores.as_ptr().add(i + 8));
            let s3 = vld1q_f32(scores.as_ptr().add(i + 12));
            let r0 = vaddq_f32(s0, bias_vec);
            let r1 = vaddq_f32(s1, bias_vec);
            let r2 = vaddq_f32(s2, bias_vec);
            let r3 = vaddq_f32(s3, bias_vec);
            vst1q_f32(scores.as_mut_ptr().add(i), r0);
            vst1q_f32(scores.as_mut_ptr().add(i + 4), r1);
            vst1q_f32(scores.as_mut_ptr().add(i + 8), r2);
            vst1q_f32(scores.as_mut_ptr().add(i + 12), r3);
            i += 16;
        }

        while i + 4 <= n {
            let s = vld1q_f32(scores.as_ptr().add(i));
            let r = vaddq_f32(s, bias_vec);
            vst1q_f32(scores.as_mut_ptr().add(i), r);
            i += 4;
        }

        while i < n {
            scores[i] += bias;
            i += 1;
        }
    }

    #[inline(always)]
    fn labels_from_argmax(&self, argmax: &[usize]) -> TractResult<Tensor> {
        if let Some(ref lbls) = self.data.labels_str {
            let mapped: Vec<String> =
                argmax.iter().map(|&i| lbls.get(i).cloned().unwrap_or_default()).collect();
            return Self::tensor1_from_vec_string(mapped);
        }
        if let Some(ref lbls) = self.data.labels_i64 {
            let mapped: Vec<i64> =
                argmax.iter().map(|&i| lbls.get(i).copied().unwrap_or(0)).collect();
            return Self::tensor1_from_vec_i64(mapped);
        }
        match self.data.labels.datum_type() {
            DatumType::String => {
                let label_slice = self.data.labels.as_slice::<String>()?;
                let mapped: Vec<String> = argmax
                    .iter()
                    .map(|&i| label_slice.get(i).cloned().unwrap_or_default())
                    .collect();
                Self::tensor1_from_vec_string(mapped)
            }
            DatumType::I64 => {
                let label_slice = self.data.labels.as_slice::<i64>()?;
                let mapped: Vec<i64> =
                    argmax.iter().map(|&i| label_slice.get(i).copied().unwrap_or(0)).collect();
                Self::tensor1_from_vec_i64(mapped)
            }
            other => bail!("Unsupported label type: {:?}", other),
        }
    }

    /// Optimized: fused matmul + bias + activation in single pass
    fn eval_scores_and_argmax_from_tensor(
        &self,
        input: &Tensor,
        want_argmax: bool,
    ) -> TractResult<(Tensor, Option<Vec<usize>>)> {
        let ishape = input.shape();
        let (n, c) = match ishape {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let e = self.n_classes();
        let eff_e = self.data.eff_e;
        let feat_c = self.data.feat_c;

        if feat_c != c {
            bail!("Feature count mismatch: expected {}, got {}", feat_c, c);
        }

        let coef_slice: &[f32] = &self.data.coefficients_raw;
        let coef_t: &[f32] = &self.data.coefficients_t;
        let packed = self.data.packed_tiled.as_ref();
        let intercepts_eff_e: Option<&[f32]> = self.data.intercepts_eff_e.as_deref();
        let bias_scalar = self.data.bias_scalar;
        let binary_intercepts_2 = self.data.binary_intercepts_2;

        let input_contig = input.as_slice::<f32>().ok();
        let input_ptr = if input_contig.is_none() { Some(input.as_ptr::<f32>()?) } else { None };
        let strides = input.strides().to_vec();
        let s0 = if n == 1 { 0 } else { strides[0] };
        let s1 = strides[ishape.len() - 1];

        // Allocate scores
        let mut scores_tensor = unsafe { Tensor::uninitialized::<f32>(&[n, eff_e]) }?;
        let scores = scores_tensor.as_slice_mut::<f32>()?;

        // Matmul phase
        unsafe {
            let use_t = self.data.prefer_transposed;
            if let Some(x) = &input_contig {
                if let Some(pk) = packed {
                    pk.gemm_rows_contig(x, n, scores);
                } else if use_t {
                    math::matmul_rows_neon_contig_t(x, n, c, coef_t, eff_e, scores);
                } else {
                    math::matmul_rows_neon_contig(x, n, c, coef_slice, eff_e, scores);
                }
            } else {
                let base = input_ptr.unwrap();
                Self::with_rowbuf(c, |rowbuf| {
                    if let Some(pk) = packed {
                        pk.gemm_rows_gather(base, n, s0, s1, scores, rowbuf);
                    } else if use_t {
                        math::matmul_rows_neon_gather_t(
                            base, n, c, s0, s1, coef_t, eff_e, scores, rowbuf,
                        );
                    } else {
                        math::matmul_rows_neon_gather(
                            base, n, c, s0, s1, coef_slice, eff_e, scores, rowbuf,
                        );
                    }
                });
            }

            // Fused bias + activation per row
            for row in scores.chunks_mut(eff_e) {
                // Add bias
                if let Some(bias_vec) = intercepts_eff_e {
                    Self::add_bias_neon(row, bias_vec);
                } else if bias_scalar != 0.0 {
                    Self::add_scalar_bias_neon(row, bias_scalar);
                }
            }

            // Apply activation
            match self.post_transform {
                PostTransformLC::None => {}
                PostTransformLC::Softmax => {
                    softmax::softmax_inplace_rows(scores, n, eff_e);
                }
                PostTransformLC::Logistic => {
                    softmax::logistic_inplace_rows(scores, n, eff_e);
                }
            }

            // Handle binary expansion
            if eff_e != e {
                debug_assert_eq!(e, 2);
                let compact = scores;
                let mut full_tensor = Tensor::uninitialized::<f32>(&[n, e])?;
                let full = full_tensor.as_slice_mut::<f32>()?;

                match self.post_transform {
                    PostTransformLC::None => {
                        for i in 0..n {
                            let v = compact[i];
                            full[i * 2] = -v;
                            full[i * 2 + 1] = v;
                        }
                    }
                    _ => {
                        for i in 0..n {
                            let v = compact[i];
                            full[i * 2] = 1.0 - v;
                            full[i * 2 + 1] = v;
                        }
                    }
                }

                if let Some([b0, b1]) = binary_intercepts_2 {
                    let bias_arr = [b0, b1];
                    for row in full.chunks_mut(2) {
                        Self::add_bias_neon(row, &bias_arr);
                    }
                }

                let argmax = if want_argmax { Some(argmax::argmax_rows(full, n, e)) } else { None };

                return Ok((full_tensor, argmax));
            }

            // Compute argmax if needed
            let argmax =
                if want_argmax { Some(argmax::argmax_rows(scores, n, eff_e)) } else { None };

            Ok((scores_tensor, argmax))
        }
    }

    pub fn eval_scores_into(
        &self,
        input: &Tensor,
        out: &mut [f32],
        want_argmax: bool,
    ) -> TractResult<(usize, usize, Option<Vec<usize>>)> {
        let ishape = input.shape();
        let (n, c) = match ishape {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let e = self.n_classes();
        let eff_e = self.data.eff_e;
        let feat_c = self.data.feat_c;

        if feat_c != c {
            bail!("Feature count mismatch: expected {}, got {}", feat_c, c);
        }

        let m_out = self.data.m_out;
        ensure!(
            out.len() == n * m_out,
            "output buffer has len {}, expected {}",
            out.len(),
            n * m_out
        );

        let coef_slice: &[f32] = &self.data.coefficients_raw;
        let coef_t: &[f32] = &self.data.coefficients_t;
        let packed = self.data.packed_tiled.as_ref();
        let intercepts_eff_e: Option<&[f32]> = self.data.intercepts_eff_e.as_deref();
        let bias_scalar = self.data.bias_scalar;
        let binary_intercepts_2 = self.data.binary_intercepts_2;

        let input_contig = input.as_slice::<f32>().ok();
        let input_ptr = if input_contig.is_none() { Some(input.as_ptr::<f32>()?) } else { None };
        let strides = input.strides().to_vec();
        let s0 = if n == 1 { 0 } else { strides[0] };
        let s1 = strides[ishape.len() - 1];

        unsafe {
            // Matmul
            if m_out == eff_e {
                let use_t = self.data.prefer_transposed;
                if let Some(x) = &input_contig {
                    if let Some(pk) = packed {
                        pk.gemm_rows_contig(x, n, out);
                    } else if use_t {
                        math::matmul_rows_neon_contig_t(x, n, c, coef_t, eff_e, out);
                    } else {
                        math::matmul_rows_neon_contig(x, n, c, coef_slice, eff_e, out);
                    }
                } else {
                    let base = input_ptr.unwrap();
                    Self::with_rowbuf(c, |rowbuf| {
                        if let Some(pk) = packed {
                            pk.gemm_rows_gather(base, n, s0, s1, out, rowbuf);
                        } else if use_t {
                            math::matmul_rows_neon_gather_t(
                                base, n, c, s0, s1, coef_t, eff_e, out, rowbuf,
                            );
                        } else {
                            math::matmul_rows_neon_gather(
                                base, n, c, s0, s1, coef_slice, eff_e, out, rowbuf,
                            );
                        }
                    });
                }

                // Fused bias addition
                for row in out.chunks_mut(eff_e) {
                    if let Some(bias_vec) = intercepts_eff_e {
                        Self::add_bias_neon(row, bias_vec);
                    } else if eff_e == 1 && bias_scalar != 0.0 {
                        Self::add_scalar_bias_neon(row, bias_scalar);
                    }
                }

                // Apply activation
                match self.post_transform {
                    PostTransformLC::None => {}
                    PostTransformLC::Softmax => {
                        softmax::softmax_inplace_rows(out, n, eff_e);
                    }
                    PostTransformLC::Logistic => {
                        softmax::logistic_inplace_rows(out, n, eff_e);
                    }
                }

                let argmax =
                    if want_argmax { Some(argmax::argmax_rows(out, n, m_out)) } else { None };

                return Ok((n, m_out, argmax));
            } else {
                // Binary compact case
                debug_assert!(eff_e == 1 && e == 2);
                let w = &coef_slice[..c];
                Self::with_compact(n, |compact| {
                    if let Some(x) = &input_contig {
                        for i in 0..n {
                            let row = &x[i * c..(i + 1) * c];
                            compact[i] = math::dot_neon(row, w);
                        }
                    } else {
                        let base = input_ptr.unwrap();
                        Self::with_rowbuf(c, |rowbuf| {
                            for i in 0..n {
                                for k in 0..c {
                                    let off = i as isize * s0 + k as isize * s1;
                                    rowbuf[k] = *base.offset(off);
                                }
                                compact[i] = math::dot_neon(&rowbuf[..c], w);
                            }
                        });
                    }

                    if bias_scalar != 0.0 {
                        Self::add_scalar_bias_neon(&mut compact[..n], bias_scalar);
                    }

                    match self.post_transform {
                        PostTransformLC::None => {}
                        PostTransformLC::Softmax | PostTransformLC::Logistic => {
                            softmax::logistic_inplace_rows(&mut compact[..n], n, 1);
                        }
                    }

                    for i in 0..n {
                        let v = compact[i];
                        match self.post_transform {
                            PostTransformLC::None => {
                                out[i * 2] = -v;
                                out[i * 2 + 1] = v;
                            }
                            _ => {
                                out[i * 2] = 1.0 - v;
                                out[i * 2 + 1] = v;
                            }
                        }
                    }

                    if let Some([b0, b1]) = binary_intercepts_2 {
                        let bias_arr = [b0, b1];
                        for row in out.chunks_mut(2) {
                            Self::add_bias_neon(row, &bias_arr);
                        }
                    }
                });

                let argmax = if want_argmax { Some(argmax::argmax_rows(out, n, 2)) } else { None };

                Ok((n, m_out, argmax))
            }
        }
    }

    pub fn eval_scores_normalized_into(
        &self,
        input: &Tensor,
        out: &mut [f32],
        want_argmax: bool,
        norm_kind: crate::ml::normalizer::NormKind,
    ) -> TractResult<(usize, usize, Option<Vec<usize>>)> {
        if self.post_transform == PostTransformLC::Softmax
            && matches!(norm_kind, crate::ml::normalizer::NormKind::L1)
        {
            return self.eval_scores_into(input, out, want_argmax);
        }
        let (n, m, argmax) = self.eval_scores_into(input, out, want_argmax)?;
        let normalizer = crate::ml::normalizer::Normalizer { kind: norm_kind };
        normalizer.eval_inplace_rows(out, n, m)?;
        Ok((n, m, argmax))
    }

    pub fn eval_from_tensor(&self, input: &Tensor) -> TractResult<(Tensor, Tensor)> {
        let (scores, argmax) = self.eval_scores_and_argmax_from_tensor(input, true)?;
        let labels = if let Some(argmax) = argmax {
            self.labels_from_argmax(&argmax)?
        } else {
            self.compute_labels(&scores)?
        };
        Ok((labels, scores))
    }

    fn compute_labels(&self, scores: &Tensor) -> TractResult<Tensor> {
        let shape = scores.shape();
        let (n, e) = match shape {
            [n, e] => (*n, *e),
            other => bail!("scores must be 2D, got {} dims", other.len()),
        };
        let data = scores.as_slice::<f32>()?;
        let argmax_indices = argmax::argmax_rows(data, n, e);

        let labels = if self.data.labels.datum_type() == DatumType::String {
            let label_slice = self.data.labels.as_slice::<String>()?;
            let mapped: Vec<String> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).cloned().unwrap_or_default())
                .collect();
            Self::tensor1_from_vec_string(mapped)?
        } else if self.data.labels.datum_type() == DatumType::I64 {
            let label_slice = self.data.labels.as_slice::<i64>()?;
            let mapped: Vec<i64> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).copied().unwrap_or(0))
                .collect();
            Self::tensor1_from_vec_i64(mapped)?
        } else {
            bail!("Unsupported label type: {:?}", self.data.labels.datum_type())
        };

        Ok(labels)
    }
}

impl Op for LinearClassifier {
    fn name(&self) -> StaticName {
        "LinearClassifier".into()
    }

    op_as_typed_op!();
}

impl EvalOp for LinearClassifier {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let (labels, scores) = self.eval_from_tensor(&input)?;
        Ok(tvec!(labels.into_tvalue(), scores.into_tvalue()))
    }
}

impl TypedOp for LinearClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        let label_dt = self.data.labels.datum_type();
        Ok(tvec!(label_dt.fact(&[n.clone()]), f32::fact(&[n.clone(), self.n_classes().into()])))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if node.outputs.len() < 2 {
            return Ok(None);
        }
        let succs = &model.node(node.id).outputs[1].successors;
        if succs.len() != 1 {
            return Ok(None);
        }
        let succ_inlet = succs[0];
        let succ_node = model.node(succ_inlet.node);
        let Some(norm) = succ_node.op_as::<crate::ml::normalizer::Normalizer>() else {
            return Ok(None);
        };

        let mut patch = TypedModelPatch::default();
        let fused_input = patch.taps(model, &[node.inputs[0]])?[0];
        let fused = PostNormalizedLinearClassifier { lc: self.clone(), norm_kind: norm.kind };
        let out =
            patch.wire_node(format!("{}._fused_lc_postnorm", node.name), fused, &[fused_input])?;
        patch.shunt_outside(model, OutletId::new(node.id, 0), out[0])?;
        patch.shunt_outside(model, OutletId::new(succ_node.id, 0), out[1])?;
        patch.dont_apply_twice = Some(format!("fuse-lc-into-post-normalizer@{}", node.id));
        Ok(Some(patch))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("labels"),
        TypeName::Scalar.tensor().named("coefficients"),
        TypeName::Scalar.tensor().named("intercepts"),
        TypeName::Integer.named("multi_class"),
        TypeName::String.named("post_transform"),
    ]
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LinearClassifier,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let labels = ast.konst_variable(format!("{}_labels", node.name), &op.data.labels)?;
    let coefficients =
        ast.konst_variable(format!("{}_coefficients", node.name), &op.data.coefficients)?;

    let post_transform_str = match op.post_transform {
        PostTransformLC::None => "NONE",
        PostTransformLC::Softmax => "SOFTMAX",
        PostTransformLC::Logistic => "LOGISTIC",
    };

    let mut args = vec![input, labels, coefficients];
    let named_args = vec![
        ("multi_class", numeric(op.multi_class as i64)),
        ("post_transform", string(post_transform_str)),
    ];

    if let Some(intercepts) = &op.data.intercepts {
        let intercepts_var = ast.konst_variable(format!("{}_intercepts", node.name), intercepts)?;
        args.push(intercepts_var);
    }

    Ok(Some(invocation("tract_onnx_ml_linear_classifier", &args, &named_args)))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let labels: Arc<Tensor> = invocation.named_arg_as(builder, "labels")?;
    let coefficients: Arc<Tensor> = invocation.named_arg_as(builder, "coefficients")?;
    let intercepts: Option<Arc<Tensor>> = invocation.named_arg_as(builder, "intercepts").ok();
    let multi_class = invocation.named_arg_as(builder, "multi_class")?;
    let post_transform: String = invocation.named_arg_as(builder, "post_transform")?;
    let post_transform = parse_post_transform(&post_transform)?;

    let e = labels.len();
    let coef_slice = coefficients.as_slice::<f32>()?;
    let coef_rank = coefficients.rank();
    let (eff_e, feat_c) = if coef_rank == 2 {
        let shape = coefficients.shape();
        (shape[0], shape[1])
    } else if coef_rank == 1 {
        if e == 2 {
            (1, coef_slice.len())
        } else if coef_slice.len() % e == 0 {
            (e, coef_slice.len() / e)
        } else {
            bail!(
                "Cannot infer (classes, features) from coefficients length {} and labels {}",
                coef_slice.len(),
                e
            )
        }
    } else {
        bail!("Unsupported coefficients rank {}", coef_rank)
    };

    if eff_e * feat_c != coef_slice.len() {
        bail!(
            "Inconsistent coefficients size: eff_e={} feat_c={} len={}",
            eff_e,
            feat_c,
            coef_slice.len()
        );
    }

    let mut transposed = Vec::<f32>::with_capacity(coef_slice.len());
    transposed.resize(coef_slice.len(), 0.0);
    for cls in 0..eff_e {
        for f in 0..feat_c {
            transposed[f * eff_e + cls] = coef_slice[cls * feat_c + f];
        }
    }
    let coefficients_raw: Arc<[f32]> = coef_slice.to_vec().into();
    let coefficients_t: Arc<[f32]> = transposed.into();

    let prefer_transposed = eff_e >= 4 && feat_c >= 16;
    let m_out = if eff_e == 1 && e == 2 { 2 } else { eff_e };
    let packed_tiled = if eff_e >= 8 && feat_c >= 16 {
        Some(MatmulTiled::new_from_transposed(&coefficients_t, feat_c, eff_e))
    } else {
        None
    };

    let binary_compact = eff_e == 1 && e == 2;
    let mut bias_scalar: f32 = 0.0;
    let mut intercepts_eff_e: Option<Arc<[f32]>> = None;
    let mut binary_intercepts_2: Option<[f32; 2]> = None;
    if let Some(b) = &intercepts {
        let bsl = b.as_slice::<f32>()?;
        if bsl.len() == 1 {
            bias_scalar = bsl[0];
        } else if bsl.len() == eff_e {
            intercepts_eff_e = Some(bsl.to_vec().into());
        } else if binary_compact && bsl.len() == 2 {
            binary_intercepts_2 = Some([bsl[0], bsl[1]]);
        }
    }

    let data = LinearClassifierData {
        labels: labels.clone(),
        coefficients: coefficients.clone(),
        intercepts: intercepts.clone(),
        eff_e,
        feat_c,
        coefficients_raw,
        coefficients_t,
        labels_i64: match labels.datum_type() {
            DatumType::I64 => Some(labels.as_slice::<i64>()?.to_vec().into()),
            _ => None,
        },
        labels_str: match labels.datum_type() {
            DatumType::String => Some(labels.as_slice::<String>()?.to_vec().into()),
            _ => None,
        },
        packed_tiled,
        prefer_transposed,
        m_out,
        binary_compact,
        bias_scalar,
        intercepts_eff_e,
        binary_intercepts_2,
    };
    let op = LinearClassifier { data, multi_class, post_transform };
    builder.wire(op, &[input])
}

#[derive(Debug, Clone, Hash)]
pub struct PostNormalizedLinearClassifier {
    pub lc: LinearClassifier,
    pub norm_kind: crate::ml::normalizer::NormKind,
}

impl Op for PostNormalizedLinearClassifier {
    fn name(&self) -> StaticName {
        "PostNormalizedLinearClassifier".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "norm={:?}, post={:?}, classes={}",
            self.norm_kind,
            self.lc.post_transform,
            self.lc.n_classes()
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for PostNormalizedLinearClassifier {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let ishape = input.shape().to_vec();
        let (outer, _in_c) = match ishape.as_slice() {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let eff_e = self.lc.data.eff_e;
        let e = self.lc.n_classes();
        let m_out = if eff_e == 1 && e == 2 { 2 } else { eff_e };
        let mut scores_tensor = unsafe { Tensor::uninitialized::<f32>(&[outer, m_out]) }?;
        let scores_buf = scores_tensor.as_slice_mut::<f32>()?;
        let (_n, _m, argmax) =
            self.lc.eval_scores_normalized_into(&input, scores_buf, true, self.norm_kind)?;
        let labels = if let Some(argmax) = argmax {
            self.lc.labels_from_argmax(&argmax)?
        } else {
            self.lc.compute_labels(&scores_tensor)?
        };
        Ok(tvec!(labels.into_tvalue(), scores_tensor.into_tvalue()))
    }
}

impl TypedOp for PostNormalizedLinearClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.lc.output_facts(inputs)
    }
    as_op!();
}
