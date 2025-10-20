// SIMD normalizer with NEON on aarch64 and AVX2 on x86_64, plus scalar fallback
#![allow(unsafe_op_in_unsafe_fn)]
use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

const SMALL_FAST_C: usize = 128;

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
        let shape = input.shape();
        let c = shape[rank - 1];
        let outer = shape[..rank - 1].iter().product::<usize>();

        let input_slice =
            input.as_slice().ok_or_else(|| format_err!("Input must be contiguous"))?;

        let mut output = vec![0.0f32; input_slice.len()];

        unsafe {
            self.eval_arch(input_slice, &mut output, outer, c);
        }

        Tensor::from_shape(shape, &output)
    }

    /// Slice-based normalization that reuses the same NEON/scalar paths as `eval`,
    /// without constructing ndarray views. `outer` is the product of leading dims,
    /// and `c` is the size of the last dimension.
    pub fn eval_into_slices(
        &self,
        input: &[f32],
        output: &mut [f32],
        outer: usize,
        c: usize,
    ) -> TractResult<()> {
        ensure!(input.len() == output.len());
        ensure!(input.len() == outer * c);
        unsafe {
            self.eval_arch(input, output, outer, c);
        }
        Ok(())
    }

    /// In-place variant: normalize rows directly in the provided buffer.
    /// The buffer must be shaped as (outer, c) in row-major order.
    pub fn eval_inplace_rows(&self, data: &mut [f32], outer: usize, c: usize) -> TractResult<()> {
        ensure!(data.len() == outer * c);
        unsafe {
            // Create a temporary immutable alias to the same memory to satisfy the API
            let input_alias = std::slice::from_raw_parts(data.as_ptr(), data.len());
            self.eval_arch(input_alias, data, outer, c);
        }
        Ok(())
    }

    // Architecture-dispatched SIMD core
    #[inline(always)]
    unsafe fn eval_arch(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            self.eval_neon(input, output, outer, c)
        }
        #[cfg(target_arch = "x86_64")]
        {
            self.eval_x86(input, output, outer, c)
        }
        #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
        {
            self.eval_scalar(input, output, outer, c)
        }
    }

    // Scalar fallback used on unsupported arches or when AVX2 unavailable
    #[inline(always)]
    unsafe fn eval_scalar(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        const EPS: f32 = 1e-12;
        for i in 0..outer {
            let offset = i * c;
            let row = &input[offset..offset + c];
            let out_row = &mut output[offset..offset + c];

            let norm = match self.kind {
                NormKind::Max => row.iter().copied().map(f32::abs).fold(0.0, f32::max),
                NormKind::L1 => row.iter().copied().map(f32::abs).sum::<f32>(),
                NormKind::L2 => row.iter().copied().map(|v| v * v).sum::<f32>(),
            };

            let scale = match self.kind {
                NormKind::L2 => 1.0 / norm.max(EPS).sqrt(),
                _ => 1.0 / norm.max(EPS),
            };
            for j in 0..c {
                out_row[j] = row[j] * scale;
            }
        }
    }

    // aarch64 NEON implementation (existing)
    #[cfg(target_arch = "aarch64")]
    unsafe fn eval_neon(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        const EPS: f32 = 1e-12;

        for i in 0..outer {
            let offset = i * c;
            let row = &input[offset..offset + c];
            let out_row = &mut output[offset..offset + c];

            if c <= SMALL_FAST_C {
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

    // x86_64 AVX2 entrypoint with runtime detection
    #[cfg(target_arch = "x86_64")]
    unsafe fn eval_x86(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        if std::is_x86_feature_detected!("avx2") {
            self.eval_avx2(input, output, outer, c)
        } else {
            self.eval_scalar(input, output, outer, c)
        }
    }

    // x86_64 AVX2 implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn eval_avx2(&self, input: &[f32], output: &mut [f32], outer: usize, c: usize) {
        const EPS: f32 = 1e-12;
        for i in 0..outer {
            let offset = i * c;
            let row = &input[offset..offset + c];
            let out_row = &mut output[offset..offset + c];

            // For simplicity and maintainability, do a two-pass SIMD approach for small sizes too
            let norm = match self.kind {
                NormKind::Max => self.compute_max_norm_avx2(row),
                NormKind::L1 => self.compute_l1_norm_avx2(row),
                NormKind::L2 => self.compute_l2_norm_avx2(row),
            };

            let scale = match self.kind {
                NormKind::L2 => 1.0 / norm.max(EPS).sqrt(),
                _ => 1.0 / norm.max(EPS),
            };

            self.scale_avx2(row, out_row, scale);
        }
    }

    // ===== Arch-specific helpers =====

    // x86_64 AVX2 helpers
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_max_norm_avx2(&self, slice: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let len = slice.len();
        let mut i = 0usize;
        
        // Use 4-way unrolling for better ILP (instruction-level parallelism)
        let mut vmax0 = _mm256_set1_ps(f32::MIN);
        let mut vmax1 = _mm256_set1_ps(f32::MIN);
        let mut vmax2 = _mm256_set1_ps(f32::MIN);
        let mut vmax3 = _mm256_set1_ps(f32::MIN);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffffu32 as i32));

        // Process 32 elements at a time (4 vectors of 8)
        while i + 32 <= len {
            let v0 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let v2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));
            
            let vabs0 = _mm256_and_ps(v0, abs_mask);
            let vabs1 = _mm256_and_ps(v1, abs_mask);
            let vabs2 = _mm256_and_ps(v2, abs_mask);
            let vabs3 = _mm256_and_ps(v3, abs_mask);
            
            vmax0 = _mm256_max_ps(vmax0, vabs0);
            vmax1 = _mm256_max_ps(vmax1, vabs1);
            vmax2 = _mm256_max_ps(vmax2, vabs2);
            vmax3 = _mm256_max_ps(vmax3, vabs3);
            i += 32;
        }
        
        // Reduce the 4 accumulators
        vmax0 = _mm256_max_ps(vmax0, vmax1);
        vmax2 = _mm256_max_ps(vmax2, vmax3);
        vmax0 = _mm256_max_ps(vmax0, vmax2);

        // Process remaining 8-element chunks
        while i + 8 <= len {
            let v = _mm256_loadu_ps(slice.as_ptr().add(i));
            let vabs = _mm256_and_ps(v, abs_mask);
            vmax0 = _mm256_max_ps(vmax0, vabs);
            i += 8;
        }

        let mut max_val = hmax256_ps(vmax0);
        
        // Scalar tail
        while i < len {
            max_val = max_val.max((*slice.get_unchecked(i)).abs());
            i += 1;
        }
        max_val
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_l1_norm_avx2(&self, slice: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let len = slice.len();
        let mut i = 0usize;
        
        // Use 4-way unrolling for better ILP
        let mut vsum0 = _mm256_setzero_ps();
        let mut vsum1 = _mm256_setzero_ps();
        let mut vsum2 = _mm256_setzero_ps();
        let mut vsum3 = _mm256_setzero_ps();
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffffu32 as i32));

        // Process 32 elements at a time (4 vectors of 8)
        while i + 32 <= len {
            let v0 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let v2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));
            
            let vabs0 = _mm256_and_ps(v0, abs_mask);
            let vabs1 = _mm256_and_ps(v1, abs_mask);
            let vabs2 = _mm256_and_ps(v2, abs_mask);
            let vabs3 = _mm256_and_ps(v3, abs_mask);
            
            vsum0 = _mm256_add_ps(vsum0, vabs0);
            vsum1 = _mm256_add_ps(vsum1, vabs1);
            vsum2 = _mm256_add_ps(vsum2, vabs2);
            vsum3 = _mm256_add_ps(vsum3, vabs3);
            i += 32;
        }
        
        // Reduce the 4 accumulators
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);

        // Process remaining 8-element chunks
        while i + 8 <= len {
            let v = _mm256_loadu_ps(slice.as_ptr().add(i));
            let vabs = _mm256_and_ps(v, abs_mask);
            vsum0 = _mm256_add_ps(vsum0, vabs);
            i += 8;
        }

        let mut sum = hsum256_ps(vsum0);
        
        // Scalar tail
        while i < len {
            sum += (*slice.get_unchecked(i)).abs();
            i += 1;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_l2_norm_avx2(&self, slice: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        if std::is_x86_feature_detected!("fma") {
            return self.compute_l2_norm_avx2_fma(slice);
        }
        let len = slice.len();
        let mut i = 0usize;
        
        // Use 4-way unrolling for better ILP
        let mut vsum0 = _mm256_setzero_ps();
        let mut vsum1 = _mm256_setzero_ps();
        let mut vsum2 = _mm256_setzero_ps();
        let mut vsum3 = _mm256_setzero_ps();
        
        // Process 32 elements at a time (4 vectors of 8)
        while i + 32 <= len {
            let v0 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let v2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));
            
            let sq0 = _mm256_mul_ps(v0, v0);
            let sq1 = _mm256_mul_ps(v1, v1);
            let sq2 = _mm256_mul_ps(v2, v2);
            let sq3 = _mm256_mul_ps(v3, v3);
            
            vsum0 = _mm256_add_ps(vsum0, sq0);
            vsum1 = _mm256_add_ps(vsum1, sq1);
            vsum2 = _mm256_add_ps(vsum2, sq2);
            vsum3 = _mm256_add_ps(vsum3, sq3);
            i += 32;
        }
        
        // Reduce the 4 accumulators
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        
        // Process remaining 8-element chunks
        while i + 8 <= len {
            let v = _mm256_loadu_ps(slice.as_ptr().add(i));
            let sq = _mm256_mul_ps(v, v);
            vsum0 = _mm256_add_ps(vsum0, sq);
            i += 8;
        }
        
        let mut sum = hsum256_ps(vsum0);
        
        // Scalar tail
        while i < len {
            let v = *slice.get_unchecked(i);
            sum += v * v;
            i += 1;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn compute_l2_norm_avx2_fma(&self, slice: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let len = slice.len();
        let mut i = 0usize;
        
        // Use 4-way unrolling for better ILP with FMA
        let mut vsum0 = _mm256_setzero_ps();
        let mut vsum1 = _mm256_setzero_ps();
        let mut vsum2 = _mm256_setzero_ps();
        let mut vsum3 = _mm256_setzero_ps();
        
        // Process 32 elements at a time (4 vectors of 8)
        while i + 32 <= len {
            let v0 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let v2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));
            
            vsum0 = _mm256_fmadd_ps(v0, v0, vsum0);
            vsum1 = _mm256_fmadd_ps(v1, v1, vsum1);
            vsum2 = _mm256_fmadd_ps(v2, v2, vsum2);
            vsum3 = _mm256_fmadd_ps(v3, v3, vsum3);
            i += 32;
        }
        
        // Reduce the 4 accumulators
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        
        // Process remaining 8-element chunks
        while i + 8 <= len {
            let v = _mm256_loadu_ps(slice.as_ptr().add(i));
            vsum0 = _mm256_fmadd_ps(v, v, vsum0);
            i += 8;
        }
        
        let mut sum = hsum256_ps(vsum0);
        
        // Scalar tail
        while i < len {
            let v = *slice.get_unchecked(i);
            sum += v * v;
            i += 1;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_avx2(&self, input: &[f32], output: &mut [f32], scale: f32) {
        use std::arch::x86_64::*;
        let len = input.len();
        let mut i = 0usize;
        let s = _mm256_set1_ps(scale);
        
        // Process 32 elements at a time (4 vectors of 8) for better throughput
        while i + 32 <= len {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i + 8));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 16));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 24));
            
            let r0 = _mm256_mul_ps(v0, s);
            let r1 = _mm256_mul_ps(v1, s);
            let r2 = _mm256_mul_ps(v2, s);
            let r3 = _mm256_mul_ps(v3, s);
            
            _mm256_storeu_ps(output.as_mut_ptr().add(i), r0);
            _mm256_storeu_ps(output.as_mut_ptr().add(i + 8), r1);
            _mm256_storeu_ps(output.as_mut_ptr().add(i + 16), r2);
            _mm256_storeu_ps(output.as_mut_ptr().add(i + 24), r3);
            i += 32;
        }
        
        // Process remaining 8-element chunks
        while i + 8 <= len {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            let r = _mm256_mul_ps(v, s);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), r);
            i += 8;
        }
        
        // Scalar tail
        while i < len {
            *output.get_unchecked_mut(i) = *input.get_unchecked(i) * scale;
            i += 1;
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe fn normalize_small_max_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = SMALL_FAST_C / 16;
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

        // Process remaining elements with 4-wide SIMD (accumulate only)
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            max = max.max(horizontal_max_f32x4(abs_v));
            i += 4;
        }

        // Scalar tail (accumulate only)
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

        // Write remainder after final scale
        let mut j = chunk_count * 16;
        while j + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(j));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(j), scaled);
            j += 4;
        }
        while j < len {
            out_row[j] = row[j] * scale;
            j += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn normalize_small_l1_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = SMALL_FAST_C / 16;
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

        // Process remaining elements with 4-wide SIMD (accumulate only)
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let abs_v = vabsq_f32(v);
            sum += horizontal_sum_f32x4(abs_v);
            i += 4;
        }

        // Scalar tail (accumulate only)
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

        // Write remainder after final scale
        let mut j = chunk_count * 16;
        while j + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(j));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(j), scaled);
            j += 4;
        }
        while j < len {
            out_row[j] = row[j] * scale;
            j += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn normalize_small_l2_neon(&self, row: &[f32], out_row: &mut [f32]) {
        const EPS: f32 = 1e-12;
        let len = row.len();
        let mut i = 0;

        const MAX_CHUNKS: usize = SMALL_FAST_C / 16;
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

        // Process remaining elements with 4-wide SIMD (accumulate only)
        if i + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let mut accum = vdupq_n_f32(0.0);
            accum = vfmaq_f32(accum, v, v);
            sum += horizontal_sum_f32x4(accum);
            i += 4;
        }

        // Scalar tail (accumulate only)
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

        // Write remainder after final scale
        let mut j = chunk_count * 16;
        while j + 4 <= len {
            let v = vld1q_f32(row.as_ptr().add(j));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(out_row.as_mut_ptr().add(j), scaled);
            j += 4;
        }
        while j < len {
            out_row[j] = row[j] * scale;
            j += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
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

    // Scalar fallbacks removed for NEON-only section; a generic scalar exists above
}

// aarch64 helpers
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
unsafe fn horizontal_sum_f32x4(v: float32x4_t) -> f32 {
    let pair_sum = vpaddq_f32(v, v);
    let final_sum = vpaddq_f32(pair_sum, pair_sum);
    vgetq_lane_f32(final_sum, 0)
}

#[cfg(target_arch = "aarch64")]
unsafe fn horizontal_max_f32x4(v: float32x4_t) -> f32 {
    let pair_max = vpmaxq_f32(v, v);
    let final_max = vpmaxq_f32(pair_max, pair_max);
    vgetq_lane_f32(final_max, 0)
}

// x86_64 helpers
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    // Efficient horizontal sum using hadd and extract
    // v = [a0, a1, a2, a3, a4, a5, a6, a7]
    let hi = _mm256_extractf128_ps(v, 1); // [a4, a5, a6, a7]
    let lo = _mm256_castps256_ps128(v); // [a0, a1, a2, a3]
    let sum_128 = _mm_add_ps(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128)); // [a0+a4+a2+a6, ...]
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x01));
    _mm_cvtss_f32(sum_32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hmax256_ps(v: __m256) -> f32 {
    // Efficient horizontal max using extract and max
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max_128 = _mm_max_ps(lo, hi);
    let max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
    let max_32 = _mm_max_ss(max_64, _mm_shuffle_ps(max_64, max_64, 0x01));
    _mm_cvtss_f32(max_32)
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
