// NEON-only math primitives used by ML ops (e.g., linear classifier)
#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: math requires target_arch = aarch64");

use std::arch::aarch64::*;

/// Horizontal sum of a float32x4 vector.
#[inline(always)]
unsafe fn horiz_sum_f32x4(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v);
    let sum = vpaddq_f32(pair, pair);
    vgetq_lane_f32(sum, 0)
}

/// NEON-accelerated dot product of two f32 slices of equal length.
#[inline(always)]
pub unsafe fn dot_neon(x: &[f32], w: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    let len = x.len();
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i + 16 <= len {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let w0 = vld1q_f32(w.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let w1 = vld1q_f32(w.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let w2 = vld1q_f32(w.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));
        let w3 = vld1q_f32(w.as_ptr().add(i + 12));
        acc0 = vfmaq_f32(acc0, x0, w0);
        acc1 = vfmaq_f32(acc1, x1, w1);
        acc2 = vfmaq_f32(acc2, x2, w2);
        acc3 = vfmaq_f32(acc3, x3, w3);
        i += 16;
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut sum = horiz_sum_f32x4(acc0);
    while i + 4 <= len {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let wv = vld1q_f32(w.as_ptr().add(i));
        let part = vfmaq_f32(vdupq_n_f32(0.0), xv, wv);
        sum += horiz_sum_f32x4(part);
        i += 4;
    }
    while i < len {
        sum += *x.get_unchecked(i) * *w.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Row-wise matrix-vector products for contiguous inputs.
#[inline(always)]
pub unsafe fn matmul_rows_neon_contig(
    input: &[f32],
    n: usize,
    c: usize,
    coef_row_major: &[f32],
    e: usize,
    out: &mut [f32],
) {
    for i in 0..n {
        let row = &input[i * c..(i + 1) * c];
        for j in 0..e {
            let w = &coef_row_major[j * c..(j + 1) * c];
            let acc = dot_neon(row, w);
            out[i * e + j] = acc;
        }
    }
}

/// Same as `matmul_rows_neon_contig`, but gathers each row with explicit strides into a temporary buffer.
#[inline(always)]
pub unsafe fn matmul_rows_neon_gather(
    input_ptr: *const f32,
    n: usize,
    c: usize,
    s0: isize,
    s1: isize,
    coef_row_major: &[f32],
    e: usize,
    out: &mut [f32],
    rowbuf: &mut [f32],
) {
    debug_assert!(rowbuf.len() >= c);
    for i in 0..n {
        for k in 0..c {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
        }
        let row = &rowbuf[..c];
        for j in 0..e {
            let w = &coef_row_major[j * c..(j + 1) * c];
            let acc = dot_neon(row, w);
            out[i * e + j] = acc;
        }
    }
}
