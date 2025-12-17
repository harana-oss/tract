#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_bias_sse42_row(row: &mut [f32], bias: &[f32]) {
    use std::arch::x86_64::*;

    log::debug!("bias: using SSE4.2 row path (len={})", row.len());
    let t = row.len();
    debug_assert_eq!(t, bias.len());
    let mut j = 0usize;

    // 16-wide (4 × 4)
    while j + 16 <= t {
        let s0 = _mm_loadu_ps(row.as_ptr().add(j));
        let s1 = _mm_loadu_ps(row.as_ptr().add(j + 4));
        let s2 = _mm_loadu_ps(row.as_ptr().add(j + 8));
        let s3 = _mm_loadu_ps(row.as_ptr().add(j + 12));
        let b0 = _mm_loadu_ps(bias.as_ptr().add(j));
        let b1 = _mm_loadu_ps(bias.as_ptr().add(j + 4));
        let b2 = _mm_loadu_ps(bias.as_ptr().add(j + 8));
        let b3 = _mm_loadu_ps(bias.as_ptr().add(j + 12));
        _mm_storeu_ps(row.as_mut_ptr().add(j), _mm_add_ps(s0, b0));
        _mm_storeu_ps(row.as_mut_ptr().add(j + 4), _mm_add_ps(s1, b1));
        _mm_storeu_ps(row.as_mut_ptr().add(j + 8), _mm_add_ps(s2, b2));
        _mm_storeu_ps(row.as_mut_ptr().add(j + 12), _mm_add_ps(s3, b3));
        j += 16;
    }

    // 4-wide
    while j + 4 <= t {
        let s = _mm_loadu_ps(row.as_ptr().add(j));
        let b = _mm_loadu_ps(bias.as_ptr().add(j));
        _mm_storeu_ps(row.as_mut_ptr().add(j), _mm_add_ps(s, b));
        j += 4;
    }

    // scalar tail
    while j < t {
        *row.get_unchecked_mut(j) += *bias.get_unchecked(j);
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_bias_avx2_row(row: &mut [f32], bias: &[f32]) {
    use std::arch::x86_64::*;

    log::debug!("bias: using AVX2 row path (len={})", row.len());
    let t = row.len();
    debug_assert_eq!(t, bias.len());
    let mut j = 0usize;

    // 32-wide (4 × 8)
    while j + 32 <= t {
        let s0 = _mm256_loadu_ps(row.as_ptr().add(j));
        let s1 = _mm256_loadu_ps(row.as_ptr().add(j + 8));
        let s2 = _mm256_loadu_ps(row.as_ptr().add(j + 16));
        let s3 = _mm256_loadu_ps(row.as_ptr().add(j + 24));
        let b0 = _mm256_loadu_ps(bias.as_ptr().add(j));
        let b1 = _mm256_loadu_ps(bias.as_ptr().add(j + 8));
        let b2 = _mm256_loadu_ps(bias.as_ptr().add(j + 16));
        let b3 = _mm256_loadu_ps(bias.as_ptr().add(j + 24));
        _mm256_storeu_ps(row.as_mut_ptr().add(j), _mm256_add_ps(s0, b0));
        _mm256_storeu_ps(row.as_mut_ptr().add(j + 8), _mm256_add_ps(s1, b1));
        _mm256_storeu_ps(row.as_mut_ptr().add(j + 16), _mm256_add_ps(s2, b2));
        _mm256_storeu_ps(row.as_mut_ptr().add(j + 24), _mm256_add_ps(s3, b3));
        j += 32;
    }

    // 8-wide
    while j + 8 <= t {
        let s = _mm256_loadu_ps(row.as_ptr().add(j));
        let b = _mm256_loadu_ps(bias.as_ptr().add(j));
        _mm256_storeu_ps(row.as_mut_ptr().add(j), _mm256_add_ps(s, b));
        j += 8;
    }

    // scalar tail
    while j < t {
        *row.get_unchecked_mut(j) += *bias.get_unchecked(j);
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_scalar_bias_sse42(out: &mut [f32], bias: f32) {
    use std::arch::x86_64::*;

    log::debug!("bias: using SSE4.2 scalar path (len={})", out.len());
    let len = out.len();
    let bias_vec = _mm_set1_ps(bias);
    let mut i = 0usize;

    while i + 16 <= len {
        let v0 = _mm_loadu_ps(out.as_ptr().add(i));
        let v1 = _mm_loadu_ps(out.as_ptr().add(i + 4));
        let v2 = _mm_loadu_ps(out.as_ptr().add(i + 8));
        let v3 = _mm_loadu_ps(out.as_ptr().add(i + 12));
        _mm_storeu_ps(out.as_mut_ptr().add(i), _mm_add_ps(v0, bias_vec));
        _mm_storeu_ps(out.as_mut_ptr().add(i + 4), _mm_add_ps(v1, bias_vec));
        _mm_storeu_ps(out.as_mut_ptr().add(i + 8), _mm_add_ps(v2, bias_vec));
        _mm_storeu_ps(out.as_mut_ptr().add(i + 12), _mm_add_ps(v3, bias_vec));
        i += 16;
    }

    while i + 4 <= len {
        let v = _mm_loadu_ps(out.as_ptr().add(i));
        _mm_storeu_ps(out.as_mut_ptr().add(i), _mm_add_ps(v, bias_vec));
        i += 4;
    }

    while i < len {
        *out.get_unchecked_mut(i) += bias;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_scalar_bias_avx2(out: &mut [f32], bias: f32) {
    use std::arch::x86_64::*;

    log::debug!("bias: using AVX2 scalar path (len={})", out.len());
    let len = out.len();
    let bias_vec = _mm256_set1_ps(bias);
    let mut i = 0usize;

    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(out.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(out.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(out.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(out.as_ptr().add(i + 24));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(v0, bias_vec));
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 8), _mm256_add_ps(v1, bias_vec));
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 16), _mm256_add_ps(v2, bias_vec));
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 24), _mm256_add_ps(v3, bias_vec));
        i += 32;
    }

    while i + 8 <= len {
        let v = _mm256_loadu_ps(out.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(v, bias_vec));
        i += 8;
    }

    while i < len {
        *out.get_unchecked_mut(i) += bias;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_bias_sse42_rows(out: &mut [f32], bias: &[f32], n: usize, t: usize) {
    log::debug!("bias: using SSE4.2 rows path (n={}, t={})", n, t);
    debug_assert_eq!(out.len(), n * t);
    debug_assert_eq!(bias.len(), t);
    for i in 0..n {
        let row = std::slice::from_raw_parts_mut(out.as_mut_ptr().add(i * t), t);
        add_bias_sse42_row(row, bias);
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn add_bias_avx2_rows(out: &mut [f32], bias: &[f32], n: usize, t: usize) {
    log::debug!("bias: using AVX2 rows path (n={}, t={})", n, t);
    debug_assert_eq!(out.len(), n * t);
    debug_assert_eq!(bias.len(), t);
    for i in 0..n {
        let row = std::slice::from_raw_parts_mut(out.as_mut_ptr().add(i * t), t);
        add_bias_avx2_row(row, bias);
    }
}

/// Add a per-target bias vector to a single row (length t).
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub unsafe fn add_bias_neon_row(row: &mut [f32], bias: &[f32]) {
    let t = row.len();
    debug_assert_eq!(t, bias.len());
    let mut j = 0usize;

    // 16-wide
    while j + 16 <= t {
        let s0 = vld1q_f32(row.as_ptr().add(j));
        let s1 = vld1q_f32(row.as_ptr().add(j + 4));
        let s2 = vld1q_f32(row.as_ptr().add(j + 8));
        let s3 = vld1q_f32(row.as_ptr().add(j + 12));
        let b0 = vld1q_f32(bias.as_ptr().add(j));
        let b1 = vld1q_f32(bias.as_ptr().add(j + 4));
        let b2 = vld1q_f32(bias.as_ptr().add(j + 8));
        let b3 = vld1q_f32(bias.as_ptr().add(j + 12));
        vst1q_f32(row.as_mut_ptr().add(j), vaddq_f32(s0, b0));
        vst1q_f32(row.as_mut_ptr().add(j + 4), vaddq_f32(s1, b1));
        vst1q_f32(row.as_mut_ptr().add(j + 8), vaddq_f32(s2, b2));
        vst1q_f32(row.as_mut_ptr().add(j + 12), vaddq_f32(s3, b3));
        j += 16;
    }

    // 4-wide
    while j + 4 <= t {
        let s = vld1q_f32(row.as_ptr().add(j));
        let b = vld1q_f32(bias.as_ptr().add(j));
        vst1q_f32(row.as_mut_ptr().add(j), vaddq_f32(s, b));
        j += 4;
    }

    // scalar tail
    while j < t {
        *row.get_unchecked_mut(j) += *bias.get_unchecked(j);
        j += 1;
    }
}

/// Add the same scalar bias to all elements of the buffer.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub unsafe fn add_scalar_bias_neon(out: &mut [f32], bias: f32) {
    let len = out.len();
    let bias_vec = vdupq_n_f32(bias);
    let mut i = 0usize;

    while i + 16 <= len {
        let v0 = vld1q_f32(out.as_ptr().add(i));
        let v1 = vld1q_f32(out.as_ptr().add(i + 4));
        let v2 = vld1q_f32(out.as_ptr().add(i + 8));
        let v3 = vld1q_f32(out.as_ptr().add(i + 12));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(v0, bias_vec));
        vst1q_f32(out.as_mut_ptr().add(i + 4), vaddq_f32(v1, bias_vec));
        vst1q_f32(out.as_mut_ptr().add(i + 8), vaddq_f32(v2, bias_vec));
        vst1q_f32(out.as_mut_ptr().add(i + 12), vaddq_f32(v3, bias_vec));
        i += 16;
    }

    while i + 4 <= len {
        let v = vld1q_f32(out.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(v, bias_vec));
        i += 4;
    }

    while i < len {
        *out.get_unchecked_mut(i) += bias;
        i += 1;
    }
}

/// Add a bias vector (length t) to n consecutive rows stored row-major in `out`.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub unsafe fn add_bias_neon_rows(out: &mut [f32], bias: &[f32], n: usize, t: usize) {
    debug_assert_eq!(out.len(), n * t);
    debug_assert_eq!(bias.len(), t);
    for i in 0..n {
        let row = std::slice::from_raw_parts_mut(out.as_mut_ptr().add(i * t), t);
        add_bias_neon_row(row, bias);
    }
}

// On x86_64, alias the NEON helpers to AVX2 implementations above.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_bias_neon_row(row: &mut [f32], bias: &[f32]) {
    if std::is_x86_feature_detected!("avx2") {
        add_bias_avx2_row(row, bias)
    } else {
        add_bias_sse42_row(row, bias)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_scalar_bias_neon(out: &mut [f32], bias: f32) {
    if std::is_x86_feature_detected!("avx2") {
        add_scalar_bias_avx2(out, bias)
    } else {
        add_scalar_bias_sse42(out, bias)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_bias_neon_rows(out: &mut [f32], bias: &[f32], n: usize, t: usize) {
    if std::is_x86_feature_detected!("avx2") {
        add_bias_avx2_rows(out, bias, n, t)
    } else {
        add_bias_sse42_rows(out, bias, n, t)
    }
}
