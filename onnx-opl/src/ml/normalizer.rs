#![allow(unsafe_op_in_unsafe_fn)]

use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
const NEON_SMALL_FAST_C: usize = 128;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_normalizer",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

pub fn parse_norm_kind(s: &str) -> TractResult<NormKind> {
    match s.to_ascii_uppercase().as_str() {
        "MAX" => Ok(NormKind::Max),
        "L1" => Ok(NormKind::L1),
        "L2" => Ok(NormKind::L2),
        other => bail!("Invalid norm kind: {}", other),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormKind {
    Max,
    L1,
    L2,
}

#[derive(Debug, Clone, Hash)]
pub struct Normalizer {
    pub kind: NormKind,
}

impl Normalizer {
    pub fn eval(&self, input: ArrayViewD<f32>) -> TractResult<Tensor> {
        let rank = input.ndim();
        ensure!(rank >= 1, "Normalizer expects rank 1 or 2 inputs");

        let shape = input.shape();
        let c = shape[rank - 1];
        let outer = shape[..rank - 1].iter().product::<usize>();

        let input_slice =
            input.as_slice().ok_or_else(|| format_err!("Input must be contiguous"))?;

        let mut output = vec![0.0f32; input_slice.len()];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.eval_neon(input_slice, &mut output, outer, c);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.eval_scalar(input_slice, &mut output, outer, c);
        }

        let output_arr = ArrayD::from_shape_vec(shape.to_vec(), output)?;
        Ok(output_arr.into_tensor())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn eval_neon(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        const EPS: f32 = 1e-12;
        const NEON_SMALL_FAST_C: usize = 128;

        for i in 0..outer {
            let offset = i * c;
            let row = &input[offset..offset + c];
            let out_row = &mut output[offset..offset + c];

            if c <= NEON_SMALL_FAST_C {
                match self.kind {
                    NormKind::Max => unsafe { self.normalize_small_max_neon(row, out_row) },
                    NormKind::L1 => unsafe { self.normalize_small_l1_neon(row, out_row) },
                    NormKind::L2 => unsafe { self.normalize_small_l2_neon(row, out_row) },
                }
                continue;
            }

            let norm = match self.kind {
                NormKind::Max => unsafe { self.compute_max_norm_neon(row) },
                NormKind::L1 => unsafe { self.compute_l1_norm_neon(row) },
                NormKind::L2 => unsafe { self.compute_l2_norm_neon(row) },
            };

            let scale = match self.kind {
                NormKind::L2 => {
                    let clamped = norm.max(EPS);
                    unsafe { self.reciprocal_sqrt_neon(clamped) }
                }
                _ => {
                    let denom = norm.max(EPS);
                    1.0 / denom
                }
            };

            unsafe { self.scale_neon(row, out_row, scale) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn normalize_small_max_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = NEON_SMALL_FAST_C / 16;
        let mut stored: [float32x4x4_t; MAX_CHUNKS] =
            [float32x4x4_t(vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                MAX_CHUNKS];
        let mut chunk_count = 0usize;

        // Process 16 elements at a time using 4-way interleaved NEON
        let mut max_vec0 = vdupq_n_f32(f32::MIN);
        let mut max_vec1 = vdupq_n_f32(f32::MIN);
        let mut max_vec2 = vdupq_n_f32(f32::MIN);
        let mut max_vec3 = vdupq_n_f32(f32::MIN);

        while i + 16 <= len {
            let v = vld1q_f32_x4(row.as_ptr().add(i));
            let abs_v0 = vabsq_f32(v.0);
            let abs_v1 = vabsq_f32(v.1);
            let abs_v2 = vabsq_f32(v.2);
            let abs_v3 = vabsq_f32(v.3);

            max_vec0 = vmaxq_f32(max_vec0, abs_v0);
            max_vec1 = vmaxq_f32(max_vec1, abs_v1);
            max_vec2 = vmaxq_f32(max_vec2, abs_v2);
            max_vec3 = vmaxq_f32(max_vec3, abs_v3);

            stored[chunk_count] = v;
            chunk_count += 1;
            i += 16;
        }

        // Reduce max across all vectors
        max_vec0 = vmaxq_f32(max_vec0, max_vec1);
        max_vec2 = vmaxq_f32(max_vec2, max_vec3);
        max_vec0 = vmaxq_f32(max_vec0, max_vec2);
        let mut max = horizontal_max_f32x4(max_vec0);

        // Handle remaining elements with 4-wide SIMD
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            max = max.max(horizontal_max_f32x4(abs_v));

            let scale = 1.0 / max.max(EPS);
            let scale_vec = vdupq_n_f32(scale);
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(i), scaled);
            i += 4;
        }

        // Scalar tail
        while i < len {
            max = max.max(row[i].abs());
            i += 1;
        }

        let scale = 1.0 / max.max(EPS);
        let scale_vec = vdupq_n_f32(scale);

        // Write back with 4-way interleaved stores
        for chunk in 0..chunk_count {
            let v = stored[chunk];
            let scaled0 = vmulq_f32(v.0, scale_vec);
            let scaled1 = vmulq_f32(v.1, scale_vec);
            let scaled2 = vmulq_f32(v.2, scale_vec);
            let scaled3 = vmulq_f32(v.3, scale_vec);
            vst1q_f32_x4(
                out_row.as_mut_ptr().add(chunk * 16),
                float32x4x4_t(scaled0, scaled1, scaled2, scaled3),
            );
        }

        let base = chunk_count * 16 + ((len - chunk_count * 16) & !3);
        for t in base..len {
            out_row[t] = row[t] * scale;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn normalize_small_l1_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = NEON_SMALL_FAST_C / 16;
        let mut stored: [float32x4x4_t; MAX_CHUNKS] =
            [float32x4x4_t(vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                MAX_CHUNKS];
        let mut chunk_count = 0usize;

        // Process 16 elements at a time using 4-way interleaved NEON
        let mut sum_vec0 = vdupq_n_f32(0.0);
        let mut sum_vec1 = vdupq_n_f32(0.0);
        let mut sum_vec2 = vdupq_n_f32(0.0);
        let mut sum_vec3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let v = vld1q_f32_x4(row.as_ptr().add(i));
            let abs_v0 = vabsq_f32(v.0);
            let abs_v1 = vabsq_f32(v.1);
            let abs_v2 = vabsq_f32(v.2);
            let abs_v3 = vabsq_f32(v.3);

            sum_vec0 = vaddq_f32(sum_vec0, abs_v0);
            sum_vec1 = vaddq_f32(sum_vec1, abs_v1);
            sum_vec2 = vaddq_f32(sum_vec2, abs_v2);
            sum_vec3 = vaddq_f32(sum_vec3, abs_v3);

            stored[chunk_count] = v;
            chunk_count += 1;
            i += 16;
        }

        // Reduce sum across all vectors
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
        sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);
        let mut sum = horizontal_sum_f32x4(sum_vec0);

        // Handle remaining elements with 4-wide SIMD
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            sum += horizontal_sum_f32x4(abs_v);

            let scale = 1.0 / sum.max(EPS);
            let scale_vec = vdupq_n_f32(scale);
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(i), scaled);
            i += 4;
        }

        // Scalar tail
        while i < len {
            sum += row[i].abs();
            i += 1;
        }

        let scale = 1.0 / sum.max(EPS);
        let scale_vec = vdupq_n_f32(scale);

        // Write back with 4-way interleaved stores
        for chunk in 0..chunk_count {
            let v = stored[chunk];
            let scaled0 = vmulq_f32(v.0, scale_vec);
            let scaled1 = vmulq_f32(v.1, scale_vec);
            let scaled2 = vmulq_f32(v.2, scale_vec);
            let scaled3 = vmulq_f32(v.3, scale_vec);
            vst1q_f32_x4(
                out_row.as_mut_ptr().add(chunk * 16),
                float32x4x4_t(scaled0, scaled1, scaled2, scaled3),
            );
        }

        let base = chunk_count * 16 + ((len - chunk_count * 16) & !3);
        for t in base..len {
            out_row[t] = row[t] * scale;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn normalize_small_l2_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = NEON_SMALL_FAST_C / 16;
        let mut stored: [float32x4x4_t; MAX_CHUNKS] =
            [float32x4x4_t(vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                MAX_CHUNKS];
        let mut chunk_count = 0usize;

        // Process 16 elements at a time using 4-way interleaved NEON with FMA
        let mut sum_vec0 = vdupq_n_f32(0.0);
        let mut sum_vec1 = vdupq_n_f32(0.0);
        let mut sum_vec2 = vdupq_n_f32(0.0);
        let mut sum_vec3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let v = vld1q_f32_x4(row.as_ptr().add(i));

            // Use FMA for better accuracy and performance
            sum_vec0 = vfmaq_f32(sum_vec0, v.0, v.0);
            sum_vec1 = vfmaq_f32(sum_vec1, v.1, v.1);
            sum_vec2 = vfmaq_f32(sum_vec2, v.2, v.2);
            sum_vec3 = vfmaq_f32(sum_vec3, v.3, v.3);

            stored[chunk_count] = v;
            chunk_count += 1;
            i += 16;
        }

        // Reduce sum across all vectors
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
        sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);
        let mut sum = horizontal_sum_f32x4(sum_vec0);

        // Handle remaining elements with 4-wide SIMD
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let mut accum = vdupq_n_f32(0.0);
            accum = vfmaq_f32(accum, v, v);
            sum += horizontal_sum_f32x4(accum);

            let scale = self.reciprocal_sqrt_neon(sum.max(EPS));
            let scale_vec = vdupq_n_f32(scale);
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(i), scaled);
            i += 4;
        }

        // Scalar tail
        while i < len {
            let val = row[i];
            sum += val * val;
            i += 1;
        }

        let scale = self.reciprocal_sqrt_neon(sum.max(EPS));
        let scale_vec = vdupq_n_f32(scale);

        // Write back with 4-way interleaved stores
        for chunk in 0..chunk_count {
            let v = stored[chunk];
            let scaled0 = vmulq_f32(v.0, scale_vec);
            let scaled1 = vmulq_f32(v.1, scale_vec);
            let scaled2 = vmulq_f32(v.2, scale_vec);
            let scaled3 = vmulq_f32(v.3, scale_vec);
            vst1q_f32_x4(
                out_row.as_mut_ptr().add(chunk * 16),
                float32x4x4_t(scaled0, scaled1, scaled2, scaled3),
            );
        }

        let base = chunk_count * 16 + ((len - chunk_count * 16) & !3);
        for t in base..len {
            out_row[t] = row[t] * scale;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn reciprocal_sqrt_neon(&self, x: f32) -> f32 {
        let v = vdupq_n_f32(x);
        let estimate = vrsqrteq_f32(v);
        // Newton-Raphson refinement for better accuracy
        let step1 = vrsqrtsq_f32(vmulq_f32(v, estimate), estimate);
        let refined1 = vmulq_f32(estimate, step1);
        // Second refinement iteration for even better precision
        let step2 = vrsqrtsq_f32(vmulq_f32(v, refined1), refined1);
        let refined2 = vmulq_f32(refined1, step2);
        vgetq_lane_f32(refined2, 0)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compute_max_norm_neon(&self, slice: &[f32]) -> f32 {
        let len = slice.len();
        let mut i = 0;

        // Process 16 elements at a time with 4-way parallelism
        let mut max_vec0 = vdupq_n_f32(f32::MIN);
        let mut max_vec1 = vdupq_n_f32(f32::MIN);
        let mut max_vec2 = vdupq_n_f32(f32::MIN);
        let mut max_vec3 = vdupq_n_f32(f32::MIN);

        while i + 16 <= len {
            let v = vld1q_f32_x4(slice.as_ptr().add(i));
            let abs_v0 = vabsq_f32(v.0);
            let abs_v1 = vabsq_f32(v.1);
            let abs_v2 = vabsq_f32(v.2);
            let abs_v3 = vabsq_f32(v.3);

            max_vec0 = vmaxq_f32(max_vec0, abs_v0);
            max_vec1 = vmaxq_f32(max_vec1, abs_v1);
            max_vec2 = vmaxq_f32(max_vec2, abs_v2);
            max_vec3 = vmaxq_f32(max_vec3, abs_v3);
            i += 16;
        }

        // Reduce max across all vectors
        max_vec0 = vmaxq_f32(max_vec0, max_vec1);
        max_vec2 = vmaxq_f32(max_vec2, max_vec3);
        max_vec0 = vmaxq_f32(max_vec0, max_vec2);

        // Process remaining 4-element chunks
        while i + 4 <= len {
            let v = vld1q_f32(slice.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            max_vec0 = vmaxq_f32(max_vec0, abs_v);
            i += 4;
        }

        let mut max = horizontal_max_f32x4(max_vec0);

        // Scalar tail
        while i < len {
            max = max.max(slice[i].abs());
            i += 1;
        }

        max
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compute_l1_norm_neon(&self, slice: &[f32]) -> f32 {
        let len = slice.len();
        let mut i = 0;

        // Process 16 elements at a time with 4-way parallelism
        let mut sum_vec0 = vdupq_n_f32(0.0);
        let mut sum_vec1 = vdupq_n_f32(0.0);
        let mut sum_vec2 = vdupq_n_f32(0.0);
        let mut sum_vec3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let v = vld1q_f32_x4(slice.as_ptr().add(i));
            let abs_v0 = vabsq_f32(v.0);
            let abs_v1 = vabsq_f32(v.1);
            let abs_v2 = vabsq_f32(v.2);
            let abs_v3 = vabsq_f32(v.3);

            sum_vec0 = vaddq_f32(sum_vec0, abs_v0);
            sum_vec1 = vaddq_f32(sum_vec1, abs_v1);
            sum_vec2 = vaddq_f32(sum_vec2, abs_v2);
            sum_vec3 = vaddq_f32(sum_vec3, abs_v3);
            i += 16;
        }

        // Reduce sum across all vectors
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
        sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);

        // Process remaining 4-element chunks
        while i + 4 <= len {
            let v = vld1q_f32(slice.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            sum_vec0 = vaddq_f32(sum_vec0, abs_v);
            i += 4;
        }

        let mut sum = horizontal_sum_f32x4(sum_vec0);

        // Scalar tail
        while i < len {
            sum += slice[i].abs();
            i += 1;
        }

        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compute_l2_norm_neon(&self, slice: &[f32]) -> f32 {
        let len = slice.len();
        let mut i = 0;

        // Process 16 elements at a time with 4-way parallelism using FMA
        let mut sum_vec0 = vdupq_n_f32(0.0);
        let mut sum_vec1 = vdupq_n_f32(0.0);
        let mut sum_vec2 = vdupq_n_f32(0.0);
        let mut sum_vec3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let v = vld1q_f32_x4(slice.as_ptr().add(i));

            // Use FMA for better accuracy and performance
            sum_vec0 = vfmaq_f32(sum_vec0, v.0, v.0);
            sum_vec1 = vfmaq_f32(sum_vec1, v.1, v.1);
            sum_vec2 = vfmaq_f32(sum_vec2, v.2, v.2);
            sum_vec3 = vfmaq_f32(sum_vec3, v.3, v.3);
            i += 16;
        }

        // Reduce sum across all vectors
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
        sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
        sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);

        // Process remaining 4-element chunks
        while i + 4 <= len {
            let v = vld1q_f32(slice.as_ptr().add(i));
            sum_vec0 = vfmaq_f32(sum_vec0, v, v);
            i += 4;
        }

        let mut sum = horizontal_sum_f32x4(sum_vec0);

        // Scalar tail
        while i < len {
            let val = slice[i];
            sum += val * val;
            i += 1;
        }

        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn scale_neon(&self, input: &[f32], output: &mut [f32], scale: f32) {
        let scale_vec = vdupq_n_f32(scale);
        let len = input.len();
        let mut i = 0;

        // Process 16 elements at a time with 4-way interleaved operations
        while i + 16 <= len {
            let v = vld1q_f32_x4(input.as_ptr().add(i));
            let result0 = vmulq_f32(v.0, scale_vec);
            let result1 = vmulq_f32(v.1, scale_vec);
            let result2 = vmulq_f32(v.2, scale_vec);
            let result3 = vmulq_f32(v.3, scale_vec);
            vst1q_f32_x4(
                output.as_mut_ptr().add(i),
                float32x4x4_t(result0, result1, result2, result3),
            );
            i += 16;
        }

        // Process remaining 4-element chunks
        while i + 4 <= len {
            let v = vld1q_f32(input.as_ptr().add(i));
            let result = vmulq_f32(v, scale_vec);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }

        // Scalar tail
        while i < len {
            output[i] = input[i] * scale;
            i += 1;
        }
    }

    fn eval_scalar(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        const EPS: f32 = 1e-12;

        for i in 0..outer {
            let offset = i * c;
            let row = &input[offset..offset + c];
            let out_row = &mut output[offset..offset + c];

            let norm = match self.kind {
                NormKind::Max => row.iter().map(|x| x.abs()).fold(f32::MIN, f32::max),
                NormKind::L1 => row.iter().map(|x| x.abs()).sum::<f32>(),
                NormKind::L2 => row.iter().map(|x| x * x).sum::<f32>(),
            };

            let scale = match self.kind {
                NormKind::L2 => 1.0 / norm.max(EPS).sqrt(),
                _ => 1.0 / norm.max(EPS),
            };

            for (o, &val) in out_row.iter_mut().zip(row.iter()) {
                *o = val * scale;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn horizontal_sum_f32x4(v: float32x4_t) -> f32 {
    let pair_sum = vpaddq_f32(v, v);
    let final_sum = vpaddq_f32(pair_sum, pair_sum);
    vgetq_lane_f32(final_sum, 0)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn horizontal_max_f32x4(v: float32x4_t) -> f32 {
    let pair_max = vpmaxq_f32(v, v);
    let final_max = vpmaxq_f32(pair_max, pair_max);
    vgetq_lane_f32(final_max, 0)
}

impl Op for Normalizer {
    fn name(&self) -> Cow<'static, str> {
        "Normalizer".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Normalizer {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input_view = input.to_array_view::<f32>()?;
        Ok(tvec!(self.eval(input_view)?.into_tvalue()))
    }
}

impl TypedOp for Normalizer {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(f32::fact(inputs[0].shape.clone())))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::String.named("norm")]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Normalizer) -> TractResult<Option<Arc<RValue>>> {
    let norm_str = match op.kind {
        NormKind::Max => "MAX",
        NormKind::L1 => "L1",
        NormKind::L2 => "L2",
    };

    let input = ast.mapping[&node.inputs[0]].clone();
    let named_args = vec![("norm", string(norm_str))];
    Ok(Some(invocation("Normalizer", &[input], &named_args)))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let norm_str: String = invocation.named_arg_as(builder, "norm")?;
    let kind = parse_norm_kind(&norm_str)?;
    let op = Normalizer { kind };
    builder.wire(op, &[input])
}
