#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: matmul requires target_arch = aarch64");

use std::arch::aarch64::*;
use std::sync::Arc;

/// Highly optimized prepacked transposed weights for NEON matmul.
/// Layout: for each feature k in [0..c), we store a contiguous row of length e_padded,
/// where e_padded = round_up(e, 16). Padding slots are zeroed.
#[derive(Debug, Clone)]
pub struct MatmulTiled {
    pub c: usize,
    pub e: usize,
    pub e_padded: usize,
    pub tile_e: usize,      // 16
    pub packed: Arc<[f32]>, // length c * e_padded
}

impl MatmulTiled {
    pub fn new_from_transposed(coef_t: &[f32], c: usize, e: usize) -> Self {
        debug_assert_eq!(coef_t.len(), c * e);
        let tile_e = 16usize;
        let e_padded = (e + tile_e - 1) / tile_e * tile_e;
        let mut buf = vec![0f32; c * e_padded];

        // Vectorized copy per row using NEON
        unsafe {
            for k in 0..c {
                let src_ptr = coef_t.as_ptr().add(k * e);
                let dst_ptr = buf.as_mut_ptr().add(k * e_padded);
                let mut j = 0usize;

                // 32-float blocks
                while j + 32 <= e {
                    let v0 = vld1q_f32(src_ptr.add(j + 0));
                    let v1 = vld1q_f32(src_ptr.add(j + 4));
                    let v2 = vld1q_f32(src_ptr.add(j + 8));
                    let v3 = vld1q_f32(src_ptr.add(j + 12));
                    let v4 = vld1q_f32(src_ptr.add(j + 16));
                    let v5 = vld1q_f32(src_ptr.add(j + 20));
                    let v6 = vld1q_f32(src_ptr.add(j + 24));
                    let v7 = vld1q_f32(src_ptr.add(j + 28));
                    vst1q_f32(dst_ptr.add(j + 0), v0);
                    vst1q_f32(dst_ptr.add(j + 4), v1);
                    vst1q_f32(dst_ptr.add(j + 8), v2);
                    vst1q_f32(dst_ptr.add(j + 12), v3);
                    vst1q_f32(dst_ptr.add(j + 16), v4);
                    vst1q_f32(dst_ptr.add(j + 20), v5);
                    vst1q_f32(dst_ptr.add(j + 24), v6);
                    vst1q_f32(dst_ptr.add(j + 28), v7);
                    j += 32;
                }

                // 16-float blocks
                while j + 16 <= e {
                    let v0 = vld1q_f32(src_ptr.add(j + 0));
                    let v1 = vld1q_f32(src_ptr.add(j + 4));
                    let v2 = vld1q_f32(src_ptr.add(j + 8));
                    let v3 = vld1q_f32(src_ptr.add(j + 12));
                    vst1q_f32(dst_ptr.add(j + 0), v0);
                    vst1q_f32(dst_ptr.add(j + 4), v1);
                    vst1q_f32(dst_ptr.add(j + 8), v2);
                    vst1q_f32(dst_ptr.add(j + 12), v3);
                    j += 16;
                }

                // 4-float blocks
                while j + 4 <= e {
                    let v0 = vld1q_f32(src_ptr.add(j));
                    vst1q_f32(dst_ptr.add(j), v0);
                    j += 4;
                }

                // Scalar tail
                while j < e {
                    *dst_ptr.add(j) = *src_ptr.add(j);
                    j += 1;
                }
            }
        }
        Self { c, e, e_padded, tile_e, packed: buf.into() }
    }

    /// High-performance contiguous GEMM with deep software pipelining and prefetching
    #[inline(always)]
    pub unsafe fn gemm_rows_contig(&self, input: &[f32], n: usize, out: &mut [f32]) {
        debug_assert!(input.len() >= n * self.c);
        debug_assert!(out.len() >= n * self.e);

        let c = self.c;
        let e = self.e;
        let ep = self.e_padded;

        for i in 0..n {
            let x_ptr = input.as_ptr().add(i * c);
            let out_ptr = out.as_mut_ptr().add(i * e);

            // Prefetch input for next row
            if i + 1 < n {
                let next_x = input.as_ptr().add((i + 1) * c);
                for pf_k in (0..c).step_by(16) {
                    std::arch::asm!(
                        "prfm pldl1keep, [{0}]",
                        in(reg) next_x.add(pf_k),
                        options(nostack, preserves_flags)
                    );
                }
            }

            let mut j = 0usize;

            // Main 16-wide loop with deep software pipelining
            while j + 16 <= e {
                // Prefetch weights ahead
                if j + 256 <= ep {
                    std::arch::asm!(
                        "prfm pldl1keep, [{0}]",
                        in(reg) self.packed.as_ptr().add(j + 256),
                        options(nostack, preserves_flags)
                    );
                }

                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                let mut k = 0usize;

                // Unroll by 8 for deep software pipelining
                while k + 8 <= c {
                    // Load input vector
                    let x0 = vld1q_f32(x_ptr.add(k + 0));
                    let x1 = vld1q_f32(x_ptr.add(k + 4));

                    // Prefetch weights for next k iteration
                    if k + 16 <= c {
                        std::arch::asm!(
                            "prfm pldl1keep, [{0}]",
                            in(reg) self.packed.as_ptr().add((k + 16) * ep + j),
                            options(nostack, preserves_flags)
                        );
                    }

                    // Process k+0
                    let w00 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 0));
                    let w01 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 4));
                    let w02 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 8));
                    let w03 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 12));
                    let x0_dup = vdupq_laneq_f32(x0, 0);
                    acc0 = vfmaq_f32(acc0, x0_dup, w00);
                    acc1 = vfmaq_f32(acc1, x0_dup, w01);
                    acc2 = vfmaq_f32(acc2, x0_dup, w02);
                    acc3 = vfmaq_f32(acc3, x0_dup, w03);

                    // Process k+1
                    let w10 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 0));
                    let w11 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 4));
                    let w12 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 8));
                    let w13 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 12));
                    let x1_dup = vdupq_laneq_f32(x0, 1);
                    acc0 = vfmaq_f32(acc0, x1_dup, w10);
                    acc1 = vfmaq_f32(acc1, x1_dup, w11);
                    acc2 = vfmaq_f32(acc2, x1_dup, w12);
                    acc3 = vfmaq_f32(acc3, x1_dup, w13);

                    // Process k+2
                    let w20 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 0));
                    let w21 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 4));
                    let w22 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 8));
                    let w23 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 12));
                    let x2_dup = vdupq_laneq_f32(x0, 2);
                    acc0 = vfmaq_f32(acc0, x2_dup, w20);
                    acc1 = vfmaq_f32(acc1, x2_dup, w21);
                    acc2 = vfmaq_f32(acc2, x2_dup, w22);
                    acc3 = vfmaq_f32(acc3, x2_dup, w23);

                    // Process k+3
                    let w30 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 0));
                    let w31 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 4));
                    let w32 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 8));
                    let w33 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 12));
                    let x3_dup = vdupq_laneq_f32(x0, 3);
                    acc0 = vfmaq_f32(acc0, x3_dup, w30);
                    acc1 = vfmaq_f32(acc1, x3_dup, w31);
                    acc2 = vfmaq_f32(acc2, x3_dup, w32);
                    acc3 = vfmaq_f32(acc3, x3_dup, w33);

                    // Process k+4
                    let w40 = vld1q_f32(self.packed.as_ptr().add((k + 4) * ep + j + 0));
                    let w41 = vld1q_f32(self.packed.as_ptr().add((k + 4) * ep + j + 4));
                    let w42 = vld1q_f32(self.packed.as_ptr().add((k + 4) * ep + j + 8));
                    let w43 = vld1q_f32(self.packed.as_ptr().add((k + 4) * ep + j + 12));
                    let x4_dup = vdupq_laneq_f32(x1, 0);
                    acc0 = vfmaq_f32(acc0, x4_dup, w40);
                    acc1 = vfmaq_f32(acc1, x4_dup, w41);
                    acc2 = vfmaq_f32(acc2, x4_dup, w42);
                    acc3 = vfmaq_f32(acc3, x4_dup, w43);

                    // Process k+5
                    let w50 = vld1q_f32(self.packed.as_ptr().add((k + 5) * ep + j + 0));
                    let w51 = vld1q_f32(self.packed.as_ptr().add((k + 5) * ep + j + 4));
                    let w52 = vld1q_f32(self.packed.as_ptr().add((k + 5) * ep + j + 8));
                    let w53 = vld1q_f32(self.packed.as_ptr().add((k + 5) * ep + j + 12));
                    let x5_dup = vdupq_laneq_f32(x1, 1);
                    acc0 = vfmaq_f32(acc0, x5_dup, w50);
                    acc1 = vfmaq_f32(acc1, x5_dup, w51);
                    acc2 = vfmaq_f32(acc2, x5_dup, w52);
                    acc3 = vfmaq_f32(acc3, x5_dup, w53);

                    // Process k+6
                    let w60 = vld1q_f32(self.packed.as_ptr().add((k + 6) * ep + j + 0));
                    let w61 = vld1q_f32(self.packed.as_ptr().add((k + 6) * ep + j + 4));
                    let w62 = vld1q_f32(self.packed.as_ptr().add((k + 6) * ep + j + 8));
                    let w63 = vld1q_f32(self.packed.as_ptr().add((k + 6) * ep + j + 12));
                    let x6_dup = vdupq_laneq_f32(x1, 2);
                    acc0 = vfmaq_f32(acc0, x6_dup, w60);
                    acc1 = vfmaq_f32(acc1, x6_dup, w61);
                    acc2 = vfmaq_f32(acc2, x6_dup, w62);
                    acc3 = vfmaq_f32(acc3, x6_dup, w63);

                    // Process k+7
                    let w70 = vld1q_f32(self.packed.as_ptr().add((k + 7) * ep + j + 0));
                    let w71 = vld1q_f32(self.packed.as_ptr().add((k + 7) * ep + j + 4));
                    let w72 = vld1q_f32(self.packed.as_ptr().add((k + 7) * ep + j + 8));
                    let w73 = vld1q_f32(self.packed.as_ptr().add((k + 7) * ep + j + 12));
                    let x7_dup = vdupq_laneq_f32(x1, 3);
                    acc0 = vfmaq_f32(acc0, x7_dup, w70);
                    acc1 = vfmaq_f32(acc1, x7_dup, w71);
                    acc2 = vfmaq_f32(acc2, x7_dup, w72);
                    acc3 = vfmaq_f32(acc3, x7_dup, w73);

                    k += 8;
                }

                // Handle remaining k with unroll-by-4
                while k + 4 <= c {
                    let x = vld1q_f32(x_ptr.add(k));

                    let w0 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 0));
                    let w1 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 0));
                    let w2 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 0));
                    let w3 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 0));

                    let w0h = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 4));
                    let w1h = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 4));
                    let w2h = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 4));
                    let w3h = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 4));

                    let w0h2 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 8));
                    let w1h2 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 8));
                    let w2h2 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 8));
                    let w3h2 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 8));

                    let w0h3 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 12));
                    let w1h3 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 12));
                    let w2h3 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 12));
                    let w3h3 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 12));

                    let x0 = vdupq_laneq_f32(x, 0);
                    let x1 = vdupq_laneq_f32(x, 1);
                    let x2 = vdupq_laneq_f32(x, 2);
                    let x3 = vdupq_laneq_f32(x, 3);

                    acc0 = vfmaq_f32(acc0, x0, w0);
                    acc0 = vfmaq_f32(acc0, x1, w1);
                    acc0 = vfmaq_f32(acc0, x2, w2);
                    acc0 = vfmaq_f32(acc0, x3, w3);

                    acc1 = vfmaq_f32(acc1, x0, w0h);
                    acc1 = vfmaq_f32(acc1, x1, w1h);
                    acc1 = vfmaq_f32(acc1, x2, w2h);
                    acc1 = vfmaq_f32(acc1, x3, w3h);

                    acc2 = vfmaq_f32(acc2, x0, w0h2);
                    acc2 = vfmaq_f32(acc2, x1, w1h2);
                    acc2 = vfmaq_f32(acc2, x2, w2h2);
                    acc2 = vfmaq_f32(acc2, x3, w3h2);

                    acc3 = vfmaq_f32(acc3, x0, w0h3);
                    acc3 = vfmaq_f32(acc3, x1, w1h3);
                    acc3 = vfmaq_f32(acc3, x2, w2h3);
                    acc3 = vfmaq_f32(acc3, x3, w3h3);

                    k += 4;
                }

                // Handle remaining k
                while k < c {
                    let xv = vdupq_n_f32(*x_ptr.add(k));
                    let wv0 = vld1q_f32(self.packed.as_ptr().add(k * ep + j + 0));
                    let wv1 = vld1q_f32(self.packed.as_ptr().add(k * ep + j + 4));
                    let wv2 = vld1q_f32(self.packed.as_ptr().add(k * ep + j + 8));
                    let wv3 = vld1q_f32(self.packed.as_ptr().add(k * ep + j + 12));
                    acc0 = vfmaq_f32(acc0, xv, wv0);
                    acc1 = vfmaq_f32(acc1, xv, wv1);
                    acc2 = vfmaq_f32(acc2, xv, wv2);
                    acc3 = vfmaq_f32(acc3, xv, wv3);
                    k += 1;
                }

                vst1q_f32(out_ptr.add(j + 0), acc0);
                vst1q_f32(out_ptr.add(j + 4), acc1);
                vst1q_f32(out_ptr.add(j + 8), acc2);
                vst1q_f32(out_ptr.add(j + 12), acc3);
                j += 16;
            }

            // 8-wide tail
            if j + 8 <= e {
                let mut acc_lo = vdupq_n_f32(0.0);
                let mut acc_hi = vdupq_n_f32(0.0);
                let mut k = 0usize;

                while k + 4 <= c {
                    let x = vld1q_f32(x_ptr.add(k));
                    let w0 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j));
                    let w1 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j));
                    let w2 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j));
                    let w3 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j));
                    let w0h = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j + 4));
                    let w1h = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j + 4));
                    let w2h = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j + 4));
                    let w3h = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j + 4));
                    let x0 = vdupq_laneq_f32(x, 0);
                    let x1 = vdupq_laneq_f32(x, 1);
                    let x2 = vdupq_laneq_f32(x, 2);
                    let x3 = vdupq_laneq_f32(x, 3);
                    acc_lo = vfmaq_f32(acc_lo, x0, w0);
                    acc_lo = vfmaq_f32(acc_lo, x1, w1);
                    acc_lo = vfmaq_f32(acc_lo, x2, w2);
                    acc_lo = vfmaq_f32(acc_lo, x3, w3);
                    acc_hi = vfmaq_f32(acc_hi, x0, w0h);
                    acc_hi = vfmaq_f32(acc_hi, x1, w1h);
                    acc_hi = vfmaq_f32(acc_hi, x2, w2h);
                    acc_hi = vfmaq_f32(acc_hi, x3, w3h);
                    k += 4;
                }

                while k < c {
                    let xv = vdupq_n_f32(*x_ptr.add(k));
                    let wv = vld1q_f32(self.packed.as_ptr().add(k * ep + j));
                    let wvh = vld1q_f32(self.packed.as_ptr().add(k * ep + j + 4));
                    acc_lo = vfmaq_f32(acc_lo, xv, wv);
                    acc_hi = vfmaq_f32(acc_hi, xv, wvh);
                    k += 1;
                }

                vst1q_f32(out_ptr.add(j), acc_lo);
                vst1q_f32(out_ptr.add(j + 4), acc_hi);
                j += 8;
            }

            // 4-wide tail
            if j + 4 <= e {
                let mut acc = vdupq_n_f32(0.0);
                let mut k = 0usize;

                while k + 4 <= c {
                    let x = vld1q_f32(x_ptr.add(k));
                    let w0 = vld1q_f32(self.packed.as_ptr().add((k + 0) * ep + j));
                    let w1 = vld1q_f32(self.packed.as_ptr().add((k + 1) * ep + j));
                    let w2 = vld1q_f32(self.packed.as_ptr().add((k + 2) * ep + j));
                    let w3 = vld1q_f32(self.packed.as_ptr().add((k + 3) * ep + j));
                    let x0 = vdupq_laneq_f32(x, 0);
                    let x1 = vdupq_laneq_f32(x, 1);
                    let x2 = vdupq_laneq_f32(x, 2);
                    let x3 = vdupq_laneq_f32(x, 3);
                    acc = vfmaq_f32(acc, x0, w0);
                    acc = vfmaq_f32(acc, x1, w1);
                    acc = vfmaq_f32(acc, x2, w2);
                    acc = vfmaq_f32(acc, x3, w3);
                    k += 4;
                }

                while k < c {
                    let xv = vdupq_n_f32(*x_ptr.add(k));
                    let wv = vld1q_f32(self.packed.as_ptr().add(k * ep + j));
                    acc = vfmaq_f32(acc, xv, wv);
                    k += 1;
                }

                vst1q_f32(out_ptr.add(j), acc);
                j += 4;
            }

            // Scalar tail for e % 4
            while j < e {
                let mut sum = 0f32;
                for k in 0..c {
                    sum += *x_ptr.add(k) * *self.packed.as_ptr().add(k * ep + j);
                }
                *out_ptr.add(j) = sum;
                j += 1;
            }
        }
    }

    /// Optimized strided gather with vectorized gathering and double-buffering
    #[inline(always)]
    pub unsafe fn gemm_rows_gather(
        &self,
        input_ptr: *const f32,
        n: usize,
        s0: isize,
        s1: isize,
        out: &mut [f32],
        rowbuf: &mut [f32],
    ) {
        debug_assert!(rowbuf.len() >= self.c * 2); // Double-buffered

        let c = self.c;
        let half_buf = c;

        // Prefill first row
        if n > 0 {
            for k in 0..c {
                let off = k as isize * s1;
                rowbuf[k] = *input_ptr.offset(off);
            }
        }

        for i in 0..n {
            let current_buf = if i % 2 == 0 { 0 } else { half_buf };
            let next_buf = if i % 2 == 0 { half_buf } else { 0 };

            // Prefetch and gather next row while computing current
            if i + 1 < n {
                let next_row_base = (i + 1) as isize * s0;

                // Vectorized gather where possible (s1 == 1 case)
                if s1 == 1 {
                    let mut k = 0;
                    while k + 8 <= c {
                        let v0 = vld1q_f32(input_ptr.offset(next_row_base + k as isize + 0));
                        let v1 = vld1q_f32(input_ptr.offset(next_row_base + k as isize + 4));
                        vst1q_f32(rowbuf.as_mut_ptr().add(next_buf + k + 0), v0);
                        vst1q_f32(rowbuf.as_mut_ptr().add(next_buf + k + 4), v1);
                        k += 8;
                    }
                    while k < c {
                        rowbuf[next_buf + k] = *input_ptr.offset(next_row_base + k as isize);
                        k += 1;
                    }
                } else {
                    // Scalar gather for strided case
                    for k in 0..c {
                        let off = next_row_base + k as isize * s1;
                        rowbuf[next_buf + k] = *input_ptr.offset(off);
                    }
                }
            }

            // Process current row
            self.gemm_rows_contig(&rowbuf[current_buf..current_buf + c], 1, &mut out[i * self.e..]);
        }
    }
}
