// NEON-only implementation
#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: softmax requires target_arch = aarch64");

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// removed prefetch stub: was a no-op

use super::smallvec::SmallVec;
thread_local! {
    static SOFTMAX_ROWBUF_TL: std::cell::RefCell<SmallVec<f32, 256>> =
        std::cell::RefCell::new(SmallVec::new());
}

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
        softmax_rows_gather_into(input_ptr, rows, cols, s0, s1, out, &mut h[..cols]);
    });
}

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

        // Vector exp approximation (poly4)
        v0 = vexp_f32_poly4(v0);
        v1 = vexp_f32_poly4(v1);
        v2 = vexp_f32_poly4(v2);
        v3 = vexp_f32_poly4(v3);

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
        v = vexp_f32_poly4(v);
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
    // FMA if available
    let prod = vmulq_f32(b, o);
    vaddq_f32(prod, two_i)
}

// Polynomial exp approximation with range reduction: exp(x) = 2^n * P(r), r in [-ln2, ln2]
// Coefficients: simple 5th-order Taylor to keep code self-contained.
// P(r) ~ c0 + c1 r + c2 r^2 + c3 r^3 + c4 r^4, with c0..c4 below.
// Implemented using vfmaq_laneq_f32 and a packed coefficient vector for c0..c3.
#[inline(always)]
unsafe fn vexp_f32_poly4(x: float32x4_t) -> float32x4_t {
    // Clamp to a safe lower bound (softmax domain is <= 0 after max-sub), avoid extreme tails
    let x = vmaxq_f32(x, vdupq_n_f32(-30.0));

    // Range reduction: n = floor(x * log2e); r = x - n*ln2
    let log2e = vdupq_n_f32(1.4426950408889634_f32);
    let ln2 = vdupq_n_f32(0.6931471805599453_f32);
    let y = vmulq_f32(x, log2e);
    // floor(y) to nearest lower int
    let n_i = vcvtq_s32_f32(vrndmq_f32(y));
    let n_f = vcvtq_f32_s32(n_i);
    let r = vsubq_f32(x, vmulq_f32(n_f, ln2));

    // Polynomial approximation P(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24
    const C0: f32 = 1.0;
    const C1: f32 = 1.0;
    const C2: f32 = 0.5;
    const C3: f32 = 1.0 / 6.0;
    const C4: f32 = 1.0 / 24.0;
    let coef = vld1q_f32([C0, C1, C2, C3].as_ptr());
    let x2 = vmulq_f32(r, r);
    let x3 = vmulq_f32(x2, r);
    let x4 = vmulq_f32(x2, x2);
    let mut p = vdupq_n_f32(0.0);
    // p = C0 + C1*r + C2*r^2 + C3*r^3 + C4*r^4
    p = vfmaq_laneq_f32(p, vdupq_n_f32(1.0), coef, 0); // + C0
    p = vfmaq_laneq_f32(p, r, coef, 1); // + C1*r
    p = vfmaq_laneq_f32(p, x2, coef, 2); // + C2*r^2
    p = vfmaq_laneq_f32(p, x3, coef, 3); // + C3*r^3
    p = vmlaq_f32(p, x4, vdupq_n_f32(C4)); // + C4*r^4

    // Scale by 2^n via exponent bit construction
    let exp_n_bits = vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23);
    let exp_n = vreinterpretq_f32_s32(exp_n_bits);
    vmulq_f32(p, exp_n)
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
            while i + 16 <= cols {
                let mut v0 = vld1q_f32(ptr.add(i));
                let mut v1 = vld1q_f32(ptr.add(i + 4));
                let mut v2 = vld1q_f32(ptr.add(i + 8));
                let mut v3 = vld1q_f32(ptr.add(i + 12));
                v0 = vnegq_f32(v0);
                v1 = vnegq_f32(v1);
                v2 = vnegq_f32(v2);
                v3 = vnegq_f32(v3);
                v0 = vexp_f32_poly4(v0);
                v1 = vexp_f32_poly4(v1);
                v2 = vexp_f32_poly4(v2);
                v3 = vexp_f32_poly4(v3);
                v0 = vaddq_f32(v0, vdupq_n_f32(1.0));
                v1 = vaddq_f32(v1, vdupq_n_f32(1.0));
                v2 = vaddq_f32(v2, vdupq_n_f32(1.0));
                v3 = vaddq_f32(v3, vdupq_n_f32(1.0));
                v0 = vrecpeq_f32(v0);
                v1 = vrecpeq_f32(v1);
                v2 = vrecpeq_f32(v2);
                v3 = vrecpeq_f32(v3);
                // One Newton-Raphson step for better reciprocal accuracy
                let two = vdupq_n_f32(2.0);
                v0 = vmulq_f32(
                    v0,
                    vsubq_f32(
                        two,
                        vmulq_f32(v0, vaddq_f32(vld1q_f32(ptr.add(i)), vdupq_n_f32(1.0))),
                    ),
                );
                v1 = vmulq_f32(
                    v1,
                    vsubq_f32(
                        two,
                        vmulq_f32(v1, vaddq_f32(vld1q_f32(ptr.add(i + 4)), vdupq_n_f32(1.0))),
                    ),
                );
                v2 = vmulq_f32(
                    v2,
                    vsubq_f32(
                        two,
                        vmulq_f32(v2, vaddq_f32(vld1q_f32(ptr.add(i + 8)), vdupq_n_f32(1.0))),
                    ),
                );
                v3 = vmulq_f32(
                    v3,
                    vsubq_f32(
                        two,
                        vmulq_f32(v3, vaddq_f32(vld1q_f32(ptr.add(i + 12)), vdupq_n_f32(1.0))),
                    ),
                );
                vst1q_f32(ptr.add(i), v0);
                vst1q_f32(ptr.add(i + 4), v1);
                vst1q_f32(ptr.add(i + 8), v2);
                vst1q_f32(ptr.add(i + 12), v3);
                i += 16;
            }
            while i + 4 <= cols {
                let mut v = vld1q_f32(ptr.add(i));
                v = vnegq_f32(v);
                v = vexp_f32_poly4(v);
                v = vaddq_f32(v, vdupq_n_f32(1.0));
                let mut y = vrecpeq_f32(v);
                let two = vdupq_n_f32(2.0);
                y = vmulq_f32(y, vsubq_f32(two, vmulq_f32(y, v)));
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
