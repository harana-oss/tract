#![allow(unsafe_op_in_unsafe_fn)]

use super::smallvec::SmallVec;

// Thread-local scratch buffer shared across arch implementations via top-level wrapper
thread_local! {
    static SOFTMAX_ROWBUF_TL: std::cell::RefCell<SmallVec<f32, 256>> =
        std::cell::RefCell::new(SmallVec::new());
}

// aarch64 (NEON) implementation
#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;

    // In-place softmax over a dense row-major 2D matrix stored in a flat slice (contiguous case).
    // "rows" is the number of rows, "cols" is the number of columns.
    #[inline(always)]
    pub fn softmax_inplace_rows(data: &mut [f32], rows: usize, cols: usize) {
        if cols <= 1 {
            return;
        }
        unsafe {
            for r in 0..rows {
                let row = &mut data[r * cols..(r + 1) * cols];
                softmax_neon(row);
            }
        }
    }

    /// In-place softmax on a single row buffer (helper for both contiguous and gathered paths).
    #[inline(always)]
    pub fn softmax_inplace_row(row: &mut [f32]) {
        if row.len() <= 1 {
            return;
        }
        unsafe {
            softmax_neon(row);
        }
    }

    /// In-place softmax over rows loaded with explicit strides into a scratch buffer, writing back to `out`.
    /// input_ptr points to the first element; strides are in elements (not bytes). s0 is row stride, s1 is inner stride.
    #[inline(always)]
    pub unsafe fn softmax_rows_gather_into(
        input_ptr: *const f32,
        rows: usize,
        cols: usize,
        s0: isize,
        s1: isize,
        out: &mut [f32],
        rowbuf: &mut [f32],
    ) {
        for r in 0..rows {
            for c in 0..cols {
                let off = r as isize * s0 + c as isize * s1;
                rowbuf[c] = *input_ptr.offset(off);
            }
            let row = &mut rowbuf[..cols];
            softmax_inplace_row(row);
            out[r * cols..(r + 1) * cols].copy_from_slice(row);
        }
    }

    /// In-place logistic over rows: y = 1 / (1 + exp(-x))
    #[inline(always)]
    pub fn logistic_inplace_rows(data: &mut [f32], rows: usize, cols: usize) {
        if cols == 0 {
            return;
        }
        debug_assert_eq!(data.len(), rows * cols);
        unsafe {
            for r in 0..rows {
                let row = &mut data[r * cols..(r + 1) * cols];
                let mut i = 0usize;
                let ptr = row.as_mut_ptr();
                // 8-way unroll
                while i + 32 <= cols {
                    let mut v0 = vld1q_f32(ptr.add(i));
                    let mut v1 = vld1q_f32(ptr.add(i + 4));
                    let mut v2 = vld1q_f32(ptr.add(i + 8));
                    let mut v3 = vld1q_f32(ptr.add(i + 12));
                    let mut v4 = vld1q_f32(ptr.add(i + 16));
                    let mut v5 = vld1q_f32(ptr.add(i + 20));
                    let mut v6 = vld1q_f32(ptr.add(i + 24));
                    let mut v7 = vld1q_f32(ptr.add(i + 28));

                    v0 = vexp_f32_softmax_fast(vnegq_f32(v0));
                    v1 = vexp_f32_softmax_fast(vnegq_f32(v1));
                    v2 = vexp_f32_softmax_fast(vnegq_f32(v2));
                    v3 = vexp_f32_softmax_fast(vnegq_f32(v3));
                    v4 = vexp_f32_softmax_fast(vnegq_f32(v4));
                    v5 = vexp_f32_softmax_fast(vnegq_f32(v5));
                    v6 = vexp_f32_softmax_fast(vnegq_f32(v6));
                    v7 = vexp_f32_softmax_fast(vnegq_f32(v7));

                    let one = vdupq_n_f32(1.0);
                    v0 = vaddq_f32(v0, one);
                    v1 = vaddq_f32(v1, one);
                    v2 = vaddq_f32(v2, one);
                    v3 = vaddq_f32(v3, one);
                    v4 = vaddq_f32(v4, one);
                    v5 = vaddq_f32(v5, one);
                    v6 = vaddq_f32(v6, one);
                    v7 = vaddq_f32(v7, one);

                    // Reciprocal with one NR refinement: y = y*(2 - d*y)
                    let mut y0 = vrecpeq_f32(v0);
                    let mut y1 = vrecpeq_f32(v1);
                    let mut y2 = vrecpeq_f32(v2);
                    let mut y3 = vrecpeq_f32(v3);
                    let mut y4 = vrecpeq_f32(v4);
                    let mut y5 = vrecpeq_f32(v5);
                    let mut y6 = vrecpeq_f32(v6);
                    let mut y7 = vrecpeq_f32(v7);
                    y0 = vmulq_f32(y0, vrecpsq_f32(y0, v0));
                    y1 = vmulq_f32(y1, vrecpsq_f32(y1, v1));
                    y2 = vmulq_f32(y2, vrecpsq_f32(y2, v2));
                    y3 = vmulq_f32(y3, vrecpsq_f32(y3, v3));
                    y4 = vmulq_f32(y4, vrecpsq_f32(y4, v4));
                    y5 = vmulq_f32(y5, vrecpsq_f32(y5, v5));
                    y6 = vmulq_f32(y6, vrecpsq_f32(y6, v6));
                    y7 = vmulq_f32(y7, vrecpsq_f32(y7, v7));

                    vst1q_f32(ptr.add(i), y0);
                    vst1q_f32(ptr.add(i + 4), y1);
                    vst1q_f32(ptr.add(i + 8), y2);
                    vst1q_f32(ptr.add(i + 12), y3);
                    vst1q_f32(ptr.add(i + 16), y4);
                    vst1q_f32(ptr.add(i + 20), y5);
                    vst1q_f32(ptr.add(i + 24), y6);
                    vst1q_f32(ptr.add(i + 28), y7);
                    i += 32;
                }
                while i + 4 <= cols {
                    let mut d = vld1q_f32(ptr.add(i));
                    d = vexp_f32_softmax_fast(vnegq_f32(d));
                    d = vaddq_f32(d, vdupq_n_f32(1.0));
                    let mut y = vrecpeq_f32(d);
                    y = vmulq_f32(y, vrecpsq_f32(y, d));
                    vst1q_f32(ptr.add(i), y);
                    i += 4;
                }
                while i < cols {
                    let z = -*ptr.add(i);
                    *ptr.add(i) = 1.0 / (1.0 + z.exp());
                    i += 1;
                }
            }
        }
    }

    // NEON softmax using vexp_f32_softmax_fast
    unsafe fn softmax_neon(row: &mut [f32]) {
        let len = row.len();
        let ptr = row.as_mut_ptr();

        // Phase 1: Find maximum with 4-way unrolling
        let mut max_vec = [
            vdupq_n_f32(f32::NEG_INFINITY),
            vdupq_n_f32(f32::NEG_INFINITY),
            vdupq_n_f32(f32::NEG_INFINITY),
            vdupq_n_f32(f32::NEG_INFINITY),
        ];
        let mut i = 0;

        while i + 16 <= len {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let v2 = vld1q_f32(ptr.add(i + 8));
            let v3 = vld1q_f32(ptr.add(i + 12));
            max_vec[0] = vmaxq_f32(max_vec[0], v0);
            max_vec[1] = vmaxq_f32(max_vec[1], v1);
            max_vec[2] = vmaxq_f32(max_vec[2], v2);
            max_vec[3] = vmaxq_f32(max_vec[3], v3);
            i += 16;
        }

        max_vec[0] = vmaxq_f32(max_vec[0], max_vec[1]);
        max_vec[2] = vmaxq_f32(max_vec[2], max_vec[3]);
        max_vec[0] = vmaxq_f32(max_vec[0], max_vec[2]);

        while i + 4 <= len {
            let v = vld1q_f32(ptr.add(i));
            max_vec[0] = vmaxq_f32(max_vec[0], v);
            i += 4;
        }

        let max_val = vmaxvq_f32(max_vec[0]);
        let mut max_scalar = max_val;

        while i < len {
            max_scalar = max_scalar.max(*ptr.add(i));
            i += 1;
        }

        // Phase 2: Compute exp and sum with 4-way unrolling
        let max_broadcast = vdupq_n_f32(max_scalar);
        let mut sum_vec = [vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        i = 0;

        while i + 16 <= len {
            let mut v0 = vld1q_f32(ptr.add(i));
            let mut v1 = vld1q_f32(ptr.add(i + 4));
            let mut v2 = vld1q_f32(ptr.add(i + 8));
            let mut v3 = vld1q_f32(ptr.add(i + 12));

            v0 = vsubq_f32(v0, max_broadcast);
            v1 = vsubq_f32(v1, max_broadcast);
            v2 = vsubq_f32(v2, max_broadcast);
            v3 = vsubq_f32(v3, max_broadcast);

            // Vector exp approximation (softmax_fast)
            v0 = vexp_f32_softmax_fast(v0);
            v1 = vexp_f32_softmax_fast(v1);
            v2 = vexp_f32_softmax_fast(v2);
            v3 = vexp_f32_softmax_fast(v3);

            vst1q_f32(ptr.add(i), v0);
            vst1q_f32(ptr.add(i + 4), v1);
            vst1q_f32(ptr.add(i + 8), v2);
            vst1q_f32(ptr.add(i + 12), v3);

            sum_vec[0] = vaddq_f32(sum_vec[0], v0);
            sum_vec[1] = vaddq_f32(sum_vec[1], v1);
            sum_vec[2] = vaddq_f32(sum_vec[2], v2);
            sum_vec[3] = vaddq_f32(sum_vec[3], v3);
            i += 16;
        }

        sum_vec[0] = vaddq_f32(sum_vec[0], sum_vec[1]);
        sum_vec[2] = vaddq_f32(sum_vec[2], sum_vec[3]);
        sum_vec[0] = vaddq_f32(sum_vec[0], sum_vec[2]);

        while i + 4 <= len {
            let mut v = vld1q_f32(ptr.add(i));
            v = vsubq_f32(v, max_broadcast);
            v = vexp_f32_softmax_fast(v);
            vst1q_f32(ptr.add(i), v);
            sum_vec[0] = vaddq_f32(sum_vec[0], v);
            i += 4;
        }

        let mut sum = vaddvq_f32(sum_vec[0]);

        while i < len {
            let v = (*ptr.add(i) - max_scalar).exp();
            *ptr.add(i) = v;
            sum += v;
            i += 1;
        }

        // Phase 3: Normalize with 4-way unrolling
        let inv_sum = 1.0 / sum;
        let inv_vec = vdupq_n_f32(inv_sum);
        i = 0;

        while i + 16 <= len {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let v2 = vld1q_f32(ptr.add(i + 8));
            let v3 = vld1q_f32(ptr.add(i + 12));

            vst1q_f32(ptr.add(i), vmulq_f32(v0, inv_vec));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(v1, inv_vec));
            vst1q_f32(ptr.add(i + 8), vmulq_f32(v2, inv_vec));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(v3, inv_vec));
            i += 16;
        }

        while i + 4 <= len {
            let v = vld1q_f32(ptr.add(i));
            vst1q_f32(ptr.add(i), vmulq_f32(v, inv_vec));
            i += 4;
        }

        while i < len {
            *ptr.add(i) *= inv_sum;
            i += 1;
        }
    }

    // --- Vector exp approximations and dispatch ---

    #[inline(always)]
    unsafe fn vexp_f32_softmax_fast(x: float32x4_t) -> float32x4_t {
        // Inputs for softmax are <= 0 after max-subtraction; clamp far tail to avoid denormals
        let x = vmaxq_f32(x, vdupq_n_f32(-30.0));

        // Initial Schraudolph-style estimate in base-e (bit construction):
        // bits = int(x * (2^23 / ln(2))) + (127 << 23)
        let scaled = vmulq_f32(x, vdupq_n_f32(12102203.2_f32));
        let bits_i = vaddq_s32(vcvtq_s32_f32(scaled), vdupq_n_s32(127 << 23));
        let b = vreinterpretq_f32_s32(bits_i);

        // Improved 2nd degree compensation without branching:
        // 3*e'' = e' * o + 2*i, where
        //   i = exponent-only (2^k), o = mantissa normalized to [1,2)
        let bits_u = vreinterpretq_u32_s32(bits_i);
        let i_bits = vandq_u32(bits_u, vdupq_n_u32(0x7f80_0000));
        let o_bits = vorrq_u32(bits_u, vdupq_n_u32(0x3f80_0000));
        let i = vreinterpretq_f32_u32(i_bits);
        let o = vreinterpretq_f32_u32(o_bits);
        let two_i = vaddq_f32(i, i);
        let prod = vmulq_f32(b, o);
        vaddq_f32(prod, two_i)
    }
}

// x86_64 (AVX2/FMA) implementation
#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn vexp_f32_softmax_fast_sse(x: __m128) -> __m128 {
        // clamp x >= -30
        let x = _mm_max_ps(x, _mm_set1_ps(-30.0));
        // scaled = x * 12102203.2f
        let scaled = _mm_mul_ps(x, _mm_set1_ps(12102203.2_f32));
        // bits_i = int(scaled) + (127 << 23)
        let mut bits_i = _mm_cvtps_epi32(scaled);
        let offset = _mm_set1_epi32(127 << 23);
        bits_i = _mm_add_epi32(bits_i, offset);
        let b = _mm_castsi128_ps(bits_i);
        // i_bits = bits & 0x7f80_0000; o_bits = bits | 0x3f80_0000
        let i_bits = _mm_and_si128(bits_i, _mm_set1_epi32(0x7f80_0000u32 as i32));
        let o_bits = _mm_or_si128(bits_i, _mm_set1_epi32(0x3f80_0000u32 as i32));
        let i = _mm_castsi128_ps(i_bits);
        let o = _mm_castsi128_ps(o_bits);
        let two_i = _mm_add_ps(i, i);
        let prod = _mm_mul_ps(b, o);
        _mm_add_ps(prod, two_i)
    }

    unsafe fn softmax_sse42(row: &mut [f32]) {
        let len = row.len();
        let ptr = row.as_mut_ptr();

        // Phase 1: max
        let mut i = 0usize;
        let mut vmax = _mm_set1_ps(f32::NEG_INFINITY);
        while i + 4 <= len {
            let v = _mm_loadu_ps(ptr.add(i));
            vmax = _mm_max_ps(vmax, v);
            i += 4;
        }
        // horizontal max to scalar
        let mut max_scalar = {
            let mut tmp = [0f32; 4];
            _mm_storeu_ps(tmp.as_mut_ptr(), vmax);
            tmp.into_iter().fold(f32::NEG_INFINITY, f32::max)
        };
        while i < len {
            max_scalar = max_scalar.max(*ptr.add(i));
            i += 1;
        }

        // Phase 2: exp and sum
        let mut sum_vec = _mm_setzero_ps();
        let vbias = _mm_set1_ps(max_scalar);
        i = 0;
        while i + 4 <= len {
            let mut v = _mm_loadu_ps(ptr.add(i));
            v = _mm_sub_ps(v, vbias);
            v = vexp_f32_softmax_fast_sse(v);
            _mm_storeu_ps(ptr.add(i), v);
            sum_vec = _mm_add_ps(sum_vec, v);
            i += 4;
        }
        let mut sum = {
            let mut tmp = [0f32; 4];
            _mm_storeu_ps(tmp.as_mut_ptr(), sum_vec);
            tmp.into_iter().sum::<f32>()
        };
        while i < len {
            let v = (*ptr.add(i) - max_scalar).exp();
            *ptr.add(i) = v;
            sum += v;
            i += 1;
        }

        // Phase 3: normalize
        let inv = 1.0 / sum;
        let vinv = _mm_set1_ps(inv);
        i = 0;
        while i + 4 <= len {
            let v = _mm_loadu_ps(ptr.add(i));
            _mm_storeu_ps(ptr.add(i), _mm_mul_ps(v, vinv));
            i += 4;
        }
        while i < len {
            *ptr.add(i) *= inv;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_inplace_rows(data: &mut [f32], rows: usize, cols: usize) {
        if std::is_x86_feature_detected!("avx2") {
            log::debug!("softmax: using AVX2 rows path (rows={}, cols={})", rows, cols);
            if cols <= 1 {
                return;
            }
            for r in 0..rows {
                let row = &mut data[r * cols..(r + 1) * cols];
                softmax_avx(row);
            }
            return;
        }
        log::debug!("softmax: using SSE4.2 rows path (rows={}, cols={})", rows, cols);
        if cols <= 1 {
            return;
        }
        for r in 0..rows {
            let row = &mut data[r * cols..(r + 1) * cols];
            softmax_sse42(row);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_inplace_row(row: &mut [f32]) {
        if std::is_x86_feature_detected!("avx2") {
            log::debug!("softmax: using AVX2 row path (len={})", row.len());
            if row.len() <= 1 {
                return;
            }
            softmax_avx(row);
            return;
        }
        log::debug!("softmax: using SSE4.2 row path (len={})", row.len());
        if row.len() <= 1 {
            return;
        }
        softmax_sse42(row);
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_rows_gather_into(
        input_ptr: *const f32,
        rows: usize,
        cols: usize,
        s0: isize,
        s1: isize,
        out: &mut [f32],
        rowbuf: &mut [f32],
    ) {
        if std::is_x86_feature_detected!("avx2") {
            log::debug!(
                "softmax: using AVX2 gather path (rows={}, cols={}, s0={}, s1={})",
                rows,
                cols,
                s0,
                s1
            );
            for r in 0..rows {
                for c in 0..cols {
                    let off = r as isize * s0 + c as isize * s1;
                    rowbuf[c] = *input_ptr.offset(off);
                }
                let row = &mut rowbuf[..cols];
                softmax_inplace_row(row);
                out[r * cols..(r + 1) * cols].copy_from_slice(row);
            }
            return;
        }
        log::debug!(
            "softmax: using SSE4.2 gather path (rows={}, cols={}, s0={}, s1={})",
            rows,
            cols,
            s0,
            s1
        );
        for r in 0..rows {
            for c in 0..cols {
                let off = r as isize * s0 + c as isize * s1;
                rowbuf[c] = *input_ptr.offset(off);
            }
            let row = &mut rowbuf[..cols];
            softmax_sse42(row);
            out[r * cols..(r + 1) * cols].copy_from_slice(row);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn logistic_inplace_rows(data: &mut [f32], rows: usize, cols: usize) {
        if cols == 0 {
            return;
        }
        debug_assert_eq!(data.len(), rows * cols);
        if std::is_x86_feature_detected!("avx2") {
            for r in 0..rows {
                let row = &mut data[r * cols..(r + 1) * cols];
                let ptr = row.as_mut_ptr();
                let mut i = 0usize;
                while i + 8 <= cols {
                    let mut v = _mm256_loadu_ps(ptr.add(i));
                    v = vexp_f32_softmax_fast_avx(_mm256_sub_ps(_mm256_setzero_ps(), v));
                    v = _mm256_add_ps(v, _mm256_set1_ps(1.0));
                    // reciprocal with NR refine: y = y*(2 - d*y)
                    let mut y = _mm256_rcp_ps(v);
                    let two = _mm256_set1_ps(2.0);
                    y = _mm256_mul_ps(y, _mm256_sub_ps(two, _mm256_mul_ps(v, y)));
                    _mm256_storeu_ps(ptr.add(i), y);
                    i += 8;
                }
                while i < cols {
                    let z = -*ptr.add(i);
                    *ptr.add(i) = 1.0 / (1.0 + z.exp());
                    i += 1;
                }
            }
            return;
        }
        for r in 0..rows {
            let row = &mut data[r * cols..(r + 1) * cols];
            let ptr = row.as_mut_ptr();
            let mut i = 0usize;
            while i + 4 <= cols {
                let mut v = _mm_loadu_ps(ptr.add(i));
                v = vexp_f32_softmax_fast_sse(_mm_sub_ps(_mm_setzero_ps(), v));
                v = _mm_add_ps(v, _mm_set1_ps(1.0));
                // reciprocal with NR refine: y = y*(2 - d*y)
                let mut y = _mm_rcp_ps(v);
                let two = _mm_set1_ps(2.0);
                y = _mm_mul_ps(y, _mm_sub_ps(two, _mm_mul_ps(v, y)));
                _mm_storeu_ps(ptr.add(i), y);
                i += 4;
            }
            while i < cols {
                let z = -*ptr.add(i);
                *ptr.add(i) = 1.0 / (1.0 + z.exp());
                i += 1;
            }
        }
    }

    // Core AVX2 softmax on a single row
    #[target_feature(enable = "avx2,fma")]
    unsafe fn softmax_avx(row: &mut [f32]) {
        let len = row.len();
        let ptr = row.as_mut_ptr();

        // Phase 1: max
        let mut i = 0usize;
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= len {
            let v = _mm256_loadu_ps(ptr.add(i));
            vmax = _mm256_max_ps(vmax, v);
            i += 8;
        }
        // horizontal max to scalar
        let mut max_scalar = {
            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), vmax);
            tmp.into_iter().fold(f32::NEG_INFINITY, f32::max)
        };
        while i < len {
            max_scalar = max_scalar.max(*ptr.add(i));
            i += 1;
        }

        // Phase 2: exp and sum
        let mut sum_vec = _mm256_setzero_ps();
        let vbias = _mm256_set1_ps(max_scalar);
        i = 0;
        while i + 8 <= len {
            let mut v = _mm256_loadu_ps(ptr.add(i));
            v = _mm256_sub_ps(v, vbias);
            v = vexp_f32_softmax_fast_avx(v);
            _mm256_storeu_ps(ptr.add(i), v);
            sum_vec = _mm256_add_ps(sum_vec, v);
            i += 8;
        }
        let mut sum = {
            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
            tmp.into_iter().sum::<f32>()
        };
        while i < len {
            let v = (*ptr.add(i) - max_scalar).exp();
            *ptr.add(i) = v;
            sum += v;
            i += 1;
        }

        // Phase 3: normalize
        let inv = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv);
        i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(ptr.add(i));
            _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, vinv));
            i += 8;
        }
        while i < len {
            *ptr.add(i) *= inv;
            i += 1;
        }
    }

    // Vector exp approximation adapted to AVX2 using bit manipulation
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vexp_f32_softmax_fast_avx(x: __m256) -> __m256 {
        // clamp x >= -30
        let x = _mm256_max_ps(x, _mm256_set1_ps(-30.0));
        // scaled = x * 12102203.2f
        let scaled = _mm256_mul_ps(x, _mm256_set1_ps(12102203.2_f32));
        // bits_i = int(scaled) + (127 << 23)
        let mut bits_i = _mm256_cvtps_epi32(scaled);
        let offset = _mm256_set1_epi32(127 << 23);
        bits_i = _mm256_add_epi32(bits_i, offset);
        let b = _mm256_castsi256_ps(bits_i);
        // i_bits = bits & 0x7f80_0000; o_bits = bits | 0x3f80_0000
        let i_bits = _mm256_and_si256(bits_i, _mm256_set1_epi32(0x7f80_0000u32 as i32));
        let o_bits = _mm256_or_si256(bits_i, _mm256_set1_epi32(0x3f80_0000u32 as i32));
        let i = _mm256_castsi256_ps(i_bits);
        let o = _mm256_castsi256_ps(o_bits);
        let two_i = _mm256_add_ps(i, i);
        let prod = _mm256_mul_ps(b, o);
        _mm256_add_ps(prod, two_i)
    }
}

// If neither aarch64 nor x86_64, fail compilation with a clear message.
#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
compile_error!("softmax is only implemented for aarch64 (NEON) and x86_64 (AVX2/FMA)");

// Re-export unified API from arch module
pub use imp::{
    logistic_inplace_rows, softmax_inplace_row, softmax_inplace_rows, softmax_rows_gather_into,
};

/// Convenience wrapper that allocates a thread-local scratch row buffer and writes results into `out`.
#[inline(always)]
pub unsafe fn softmax_rows_gather_into_tl(
    input_ptr: *const f32,
    rows: usize,
    cols: usize,
    s0: isize,
    s1: isize,
    out: &mut [f32],
) {
    SOFTMAX_ROWBUF_TL.with(|rb| {
        let mut v = rb.borrow_mut();
        let mut h = v.handle();
        if h.len() < cols {
            h.resize(cols, 0.0);
        }
        // Delegate to arch-specific implementation
        softmax_rows_gather_into(input_ptr, rows, cols, s0, s1, out, &mut h[..cols]);
    });
}

// (Removed auxiliary benchmarking wrappers and alternate exp/softmax implementations)
