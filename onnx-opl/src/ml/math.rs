#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: math requires target_arch = aarch64");

use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_avx(__m256: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(__m256, 1);
    let lo = _mm256_castps256_ps128(__m256);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_avx2(x: &[f32], w: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    let len = x.len();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i + 0));
        let w0 = _mm256_loadu_ps(w.as_ptr().add(i + 0));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let w1 = _mm256_loadu_ps(w.as_ptr().add(i + 8));
        let x2 = _mm256_loadu_ps(x.as_ptr().add(i + 16));
        let w2 = _mm256_loadu_ps(w.as_ptr().add(i + 16));
        let x3 = _mm256_loadu_ps(x.as_ptr().add(i + 24));
        let w3 = _mm256_loadu_ps(w.as_ptr().add(i + 24));

        acc0 = _mm256_fmadd_ps(x0, w0, acc0);
        acc1 = _mm256_fmadd_ps(x1, w1, acc1);
        acc2 = _mm256_fmadd_ps(x2, w2, acc2);
        acc3 = _mm256_fmadd_ps(x3, w3, acc3);
        i += 32;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    let mut sum = hsum_avx(acc0);

    while i + 8 <= len {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let part = _mm256_mul_ps(xv, wv);
        sum += hsum_avx(part);
        i += 8;
    }
    while i < len {
        sum += *x.get_unchecked(i) * *w.get_unchecked(i);
        i += 1;
    }
    sum
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn prefetch_read_l1(ptr: *const f32, _bytes_ahead: isize) {
    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_rows_avx2_contig(
    input: &[f32],
    n: usize,
    c: usize,
    coef_row_major: &[f32],
    e: usize,
    out: &mut [f32],
) {
    debug_assert!(input.len() >= n * c);
    debug_assert!(coef_row_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    for i in 0..n {
        let row_ptr = input.as_ptr().add(i * c);
        let out_ptr = out.as_mut_ptr().add(i * e);

        let mut j = 0usize;
        while j + 4 <= e {
            let w0_ptr = coef_row_major.as_ptr().add((j + 0) * c);
            let w1_ptr = coef_row_major.as_ptr().add((j + 1) * c);
            let w2_ptr = coef_row_major.as_ptr().add((j + 2) * c);
            let w3_ptr = coef_row_major.as_ptr().add((j + 3) * c);

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut k = 0usize;
            while k + 32 <= c {
                let x0 = _mm256_loadu_ps(row_ptr.add(k + 0));
                let x1 = _mm256_loadu_ps(row_ptr.add(k + 8));
                let x2 = _mm256_loadu_ps(row_ptr.add(k + 16));
                let x3 = _mm256_loadu_ps(row_ptr.add(k + 24));

                let w00 = _mm256_loadu_ps(w0_ptr.add(k + 0));
                let w01 = _mm256_loadu_ps(w0_ptr.add(k + 8));
                let w02 = _mm256_loadu_ps(w0_ptr.add(k + 16));
                let w03 = _mm256_loadu_ps(w0_ptr.add(k + 24));
                acc0 = _mm256_fmadd_ps(x0, w00, acc0);
                acc0 = _mm256_fmadd_ps(x1, w01, acc0);
                acc0 = _mm256_fmadd_ps(x2, w02, acc0);
                acc0 = _mm256_fmadd_ps(x3, w03, acc0);

                let w10 = _mm256_loadu_ps(w1_ptr.add(k + 0));
                let w11 = _mm256_loadu_ps(w1_ptr.add(k + 8));
                let w12 = _mm256_loadu_ps(w1_ptr.add(k + 16));
                let w13 = _mm256_loadu_ps(w1_ptr.add(k + 24));
                acc1 = _mm256_fmadd_ps(x0, w10, acc1);
                acc1 = _mm256_fmadd_ps(x1, w11, acc1);
                acc1 = _mm256_fmadd_ps(x2, w12, acc1);
                acc1 = _mm256_fmadd_ps(x3, w13, acc1);

                let w20 = _mm256_loadu_ps(w2_ptr.add(k + 0));
                let w21 = _mm256_loadu_ps(w2_ptr.add(k + 8));
                let w22 = _mm256_loadu_ps(w2_ptr.add(k + 16));
                let w23 = _mm256_loadu_ps(w2_ptr.add(k + 24));
                acc2 = _mm256_fmadd_ps(x0, w20, acc2);
                acc2 = _mm256_fmadd_ps(x1, w21, acc2);
                acc2 = _mm256_fmadd_ps(x2, w22, acc2);
                acc2 = _mm256_fmadd_ps(x3, w23, acc2);

                let w30 = _mm256_loadu_ps(w3_ptr.add(k + 0));
                let w31 = _mm256_loadu_ps(w3_ptr.add(k + 8));
                let w32 = _mm256_loadu_ps(w3_ptr.add(k + 16));
                let w33 = _mm256_loadu_ps(w3_ptr.add(k + 24));
                acc3 = _mm256_fmadd_ps(x0, w30, acc3);
                acc3 = _mm256_fmadd_ps(x1, w31, acc3);
                acc3 = _mm256_fmadd_ps(x2, w32, acc3);
                acc3 = _mm256_fmadd_ps(x3, w33, acc3);

                k += 32;
            }

            while k + 8 <= c {
                let x = _mm256_loadu_ps(row_ptr.add(k));
                let w0 = _mm256_loadu_ps(w0_ptr.add(k));
                let w1 = _mm256_loadu_ps(w1_ptr.add(k));
                let w2 = _mm256_loadu_ps(w2_ptr.add(k));
                let w3 = _mm256_loadu_ps(w3_ptr.add(k));
                acc0 = _mm256_fmadd_ps(x, w0, acc0);
                acc1 = _mm256_fmadd_ps(x, w1, acc1);
                acc2 = _mm256_fmadd_ps(x, w2, acc2);
                acc3 = _mm256_fmadd_ps(x, w3, acc3);
                k += 8;
            }

            let mut s0 = hsum_avx(acc0);
            let mut s1 = hsum_avx(acc1);
            let mut s2 = hsum_avx(acc2);
            let mut s3 = hsum_avx(acc3);
            while k < c {
                let xv = *row_ptr.add(k);
                s0 += xv * *w0_ptr.add(k);
                s1 += xv * *w1_ptr.add(k);
                s2 += xv * *w2_ptr.add(k);
                s3 += xv * *w3_ptr.add(k);
                k += 1;
            }

            *out_ptr.add(j + 0) = s0;
            *out_ptr.add(j + 1) = s1;
            *out_ptr.add(j + 2) = s2;
            *out_ptr.add(j + 3) = s3;

            j += 4;
        }

        while j < e {
            let w_ptr = coef_row_major.as_ptr().add(j * c);
            let mut accv = _mm256_setzero_ps();
            let mut k = 0usize;
            while k + 32 <= c {
                let x0 = _mm256_loadu_ps(row_ptr.add(k + 0));
                let w0 = _mm256_loadu_ps(w_ptr.add(k + 0));
                let x1 = _mm256_loadu_ps(row_ptr.add(k + 8));
                let w1 = _mm256_loadu_ps(w_ptr.add(k + 8));
                let x2 = _mm256_loadu_ps(row_ptr.add(k + 16));
                let w2 = _mm256_loadu_ps(w_ptr.add(k + 16));
                let x3 = _mm256_loadu_ps(row_ptr.add(k + 24));
                let w3 = _mm256_loadu_ps(w_ptr.add(k + 24));
                accv = _mm256_fmadd_ps(x0, w0, accv);
                accv = _mm256_fmadd_ps(x1, w1, accv);
                accv = _mm256_fmadd_ps(x2, w2, accv);
                accv = _mm256_fmadd_ps(x3, w3, accv);
                k += 32;
            }
            while k + 8 <= c {
                let x = _mm256_loadu_ps(row_ptr.add(k));
                let w = _mm256_loadu_ps(w_ptr.add(k));
                accv = _mm256_fmadd_ps(x, w, accv);
                k += 8;
            }
            let mut sum = hsum_avx(accv);
            while k < c {
                sum += *row_ptr.add(k) * *w_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_rows_avx2_contig_t(
    input: &[f32],
    n: usize,
    c: usize,
    coef_col_major: &[f32],
    e: usize,
    out: &mut [f32],
) {
    debug_assert!(input.len() >= n * c);
    debug_assert!(coef_col_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    for i in 0..n {
        let row_ptr = input.as_ptr().add(i * c);
        let out_ptr = out.as_mut_ptr().add(i * e);

        let mut j = 0usize;
        while j + 8 <= e {
            let mut acc = _mm256_setzero_ps();

            let mut k = 0usize;
            while k + 4 <= c {
                prefetch_read_l1(row_ptr.add(k + 32), 128);

                let x0 = _mm256_broadcast_ss(row_ptr.add(k + 0));
                let x1 = _mm256_broadcast_ss(row_ptr.add(k + 1));
                let x2 = _mm256_broadcast_ss(row_ptr.add(k + 2));
                let x3 = _mm256_broadcast_ss(row_ptr.add(k + 3));

                let w0 = _mm256_loadu_ps(coef_col_major.as_ptr().add((k + 0) * e + j));
                let w1 = _mm256_loadu_ps(coef_col_major.as_ptr().add((k + 1) * e + j));
                let w2 = _mm256_loadu_ps(coef_col_major.as_ptr().add((k + 2) * e + j));
                let w3 = _mm256_loadu_ps(coef_col_major.as_ptr().add((k + 3) * e + j));

                acc = _mm256_fmadd_ps(x0, w0, acc);
                acc = _mm256_fmadd_ps(x1, w1, acc);
                acc = _mm256_fmadd_ps(x2, w2, acc);
                acc = _mm256_fmadd_ps(x3, w3, acc);
                k += 4;
            }
            while k < c {
                let xv = _mm256_broadcast_ss(row_ptr.add(k));
                let wv = _mm256_loadu_ps(coef_col_major.as_ptr().add(k * e + j));
                acc = _mm256_fmadd_ps(xv, wv, acc);
                k += 1;
            }
            _mm256_storeu_ps(out_ptr.add(j), acc);
            j += 8;
        }

        while j < e {
            let mut sum = 0f32;
            let mut k = 0usize;
            while k + 8 <= c {
                let x = _mm256_loadu_ps(row_ptr.add(k));
                let w = _mm256_loadu_ps(coef_col_major.as_ptr().add(k * e + j));
                let part = _mm256_mul_ps(x, w);
                sum += hsum_avx(part);
                k += 8;
            }
            while k < c {
                sum += *row_ptr.add(k) * *coef_col_major.as_ptr().add(k * e + j);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_rows_avx2_gather_t(
    input_ptr: *const f32,
    n: usize,
    c: usize,
    s0: isize,
    s1: isize,
    coef_col_major: &[f32],
    e: usize,
    out: &mut [f32],
    rowbuf: &mut [f32],
) {
    debug_assert!(rowbuf.len() >= c);
    debug_assert!(coef_col_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    if s1 == 1 && s0 == c as isize {
        let input_slice = std::slice::from_raw_parts(input_ptr, n * c);
        return matmul_rows_avx2_contig_t(input_slice, n, c, coef_col_major, e, out);
    }

    for i in 0..n {
        let mut k = 0usize;
        while k + 8 <= c {
            let base = i as isize * s0 + k as isize * s1;
            rowbuf[k + 0] = *input_ptr.offset(base + 0 * s1);
            rowbuf[k + 1] = *input_ptr.offset(base + 1 * s1);
            rowbuf[k + 2] = *input_ptr.offset(base + 2 * s1);
            rowbuf[k + 3] = *input_ptr.offset(base + 3 * s1);
            rowbuf[k + 4] = *input_ptr.offset(base + 4 * s1);
            rowbuf[k + 5] = *input_ptr.offset(base + 5 * s1);
            rowbuf[k + 6] = *input_ptr.offset(base + 6 * s1);
            rowbuf[k + 7] = *input_ptr.offset(base + 7 * s1);
            k += 8;
        }
        while k < c {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
            k += 1;
        }

        matmul_rows_avx2_contig_t(&rowbuf[..c], 1, c, coef_col_major, e, &mut out[i * e..]);
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn matmul_rows_avx2_gather(
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
    debug_assert!(coef_row_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    if s1 == 1 && s0 == c as isize {
        let input_slice = std::slice::from_raw_parts(input_ptr, n * c);
        return matmul_rows_avx2_contig(input_slice, n, c, coef_row_major, e, out);
    }

    for i in 0..n {
        let mut k = 0usize;
        while k + 8 <= c {
            let base = i as isize * s0 + k as isize * s1;
            rowbuf[k + 0] = *input_ptr.offset(base + 0 * s1);
            rowbuf[k + 1] = *input_ptr.offset(base + 1 * s1);
            rowbuf[k + 2] = *input_ptr.offset(base + 2 * s1);
            rowbuf[k + 3] = *input_ptr.offset(base + 3 * s1);
            rowbuf[k + 4] = *input_ptr.offset(base + 4 * s1);
            rowbuf[k + 5] = *input_ptr.offset(base + 5 * s1);
            rowbuf[k + 6] = *input_ptr.offset(base + 6 * s1);
            rowbuf[k + 7] = *input_ptr.offset(base + 7 * s1);
            k += 8;
        }
        while k < c {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
            k += 1;
        }

        let row_ptr = rowbuf.as_ptr();
        let out_ptr = out.as_mut_ptr().add(i * e);

        let mut j = 0usize;
        while j + 4 <= e {
            let w0_ptr = coef_row_major.as_ptr().add((j + 0) * c);
            let w1_ptr = coef_row_major.as_ptr().add((j + 1) * c);
            let w2_ptr = coef_row_major.as_ptr().add((j + 2) * c);
            let w3_ptr = coef_row_major.as_ptr().add((j + 3) * c);

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut k = 0usize;
            while k + 32 <= c {
                let x0 = _mm256_loadu_ps(row_ptr.add(k + 0));
                let x1 = _mm256_loadu_ps(row_ptr.add(k + 8));
                let x2 = _mm256_loadu_ps(row_ptr.add(k + 16));
                let x3 = _mm256_loadu_ps(row_ptr.add(k + 24));

                let w00 = _mm256_loadu_ps(w0_ptr.add(k + 0));
                let w01 = _mm256_loadu_ps(w0_ptr.add(k + 8));
                let w02 = _mm256_loadu_ps(w0_ptr.add(k + 16));
                let w03 = _mm256_loadu_ps(w0_ptr.add(k + 24));
                acc0 = _mm256_fmadd_ps(x0, w00, acc0);
                acc0 = _mm256_fmadd_ps(x1, w01, acc0);
                acc0 = _mm256_fmadd_ps(x2, w02, acc0);
                acc0 = _mm256_fmadd_ps(x3, w03, acc0);

                let w10 = _mm256_loadu_ps(w1_ptr.add(k + 0));
                let w11 = _mm256_loadu_ps(w1_ptr.add(k + 8));
                let w12 = _mm256_loadu_ps(w1_ptr.add(k + 16));
                let w13 = _mm256_loadu_ps(w1_ptr.add(k + 24));
                acc1 = _mm256_fmadd_ps(x0, w10, acc1);
                acc1 = _mm256_fmadd_ps(x1, w11, acc1);
                acc1 = _mm256_fmadd_ps(x2, w12, acc1);
                acc1 = _mm256_fmadd_ps(x3, w13, acc1);

                let w20 = _mm256_loadu_ps(w2_ptr.add(k + 0));
                let w21 = _mm256_loadu_ps(w2_ptr.add(k + 8));
                let w22 = _mm256_loadu_ps(w2_ptr.add(k + 16));
                let w23 = _mm256_loadu_ps(w2_ptr.add(k + 24));
                acc2 = _mm256_fmadd_ps(x0, w20, acc2);
                acc2 = _mm256_fmadd_ps(x1, w21, acc2);
                acc2 = _mm256_fmadd_ps(x2, w22, acc2);
                acc2 = _mm256_fmadd_ps(x3, w23, acc2);

                let w30 = _mm256_loadu_ps(w3_ptr.add(k + 0));
                let w31 = _mm256_loadu_ps(w3_ptr.add(k + 8));
                let w32 = _mm256_loadu_ps(w3_ptr.add(k + 16));
                let w33 = _mm256_loadu_ps(w3_ptr.add(k + 24));
                acc3 = _mm256_fmadd_ps(x0, w30, acc3);
                acc3 = _mm256_fmadd_ps(x1, w31, acc3);
                acc3 = _mm256_fmadd_ps(x2, w32, acc3);
                acc3 = _mm256_fmadd_ps(x3, w33, acc3);

                k += 32;
            }
            while k + 8 <= c {
                let x = _mm256_loadu_ps(row_ptr.add(k));
                let w0 = _mm256_loadu_ps(w0_ptr.add(k));
                let w1 = _mm256_loadu_ps(w1_ptr.add(k));
                let w2 = _mm256_loadu_ps(w2_ptr.add(k));
                let w3 = _mm256_loadu_ps(w3_ptr.add(k));
                acc0 = _mm256_fmadd_ps(x, w0, acc0);
                acc1 = _mm256_fmadd_ps(x, w1, acc1);
                acc2 = _mm256_fmadd_ps(x, w2, acc2);
                acc3 = _mm256_fmadd_ps(x, w3, acc3);
                k += 8;
            }
            let mut s0 = hsum_avx(acc0);
            let mut s1 = hsum_avx(acc1);
            let mut s2 = hsum_avx(acc2);
            let mut s3 = hsum_avx(acc3);
            while k < c {
                let xv = *row_ptr.add(k);
                s0 += xv * *w0_ptr.add(k);
                s1 += xv * *w1_ptr.add(k);
                s2 += xv * *w2_ptr.add(k);
                s3 += xv * *w3_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j + 0) = s0;
            *out_ptr.add(j + 1) = s1;
            *out_ptr.add(j + 2) = s2;
            *out_ptr.add(j + 3) = s3;
            j += 4;
        }

        while j < e {
            let w_ptr = coef_row_major.as_ptr().add(j * c);
            let mut accv = _mm256_setzero_ps();
            let mut k = 0usize;
            while k + 32 <= c {
                let x0 = _mm256_loadu_ps(row_ptr.add(k + 0));
                let w0 = _mm256_loadu_ps(w_ptr.add(k + 0));
                let x1 = _mm256_loadu_ps(row_ptr.add(k + 8));
                let w1 = _mm256_loadu_ps(w_ptr.add(k + 8));
                let x2 = _mm256_loadu_ps(row_ptr.add(k + 16));
                let w2 = _mm256_loadu_ps(w_ptr.add(k + 16));
                let x3 = _mm256_loadu_ps(row_ptr.add(k + 24));
                let w3 = _mm256_loadu_ps(w_ptr.add(k + 24));
                accv = _mm256_fmadd_ps(x0, w0, accv);
                accv = _mm256_fmadd_ps(x1, w1, accv);
                accv = _mm256_fmadd_ps(x2, w2, accv);
                accv = _mm256_fmadd_ps(x3, w3, accv);
                k += 32;
            }
            while k + 8 <= c {
                let x = _mm256_loadu_ps(row_ptr.add(k));
                let w = _mm256_loadu_ps(w_ptr.add(k));
                accv = _mm256_fmadd_ps(x, w, accv);
                k += 8;
            }
            let mut sum = hsum_avx(accv);
            while k < c {
                sum += *row_ptr.add(k) * *w_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}

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
        let x0 = vld1q_f32(x.as_ptr().add(i + 0));
        let w0 = vld1q_f32(w.as_ptr().add(i + 0));
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

    // Reduce 4 accumulators
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    let mut sum = vaddvq_f32(acc0);

    while i + 4 <= len {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let wv = vld1q_f32(w.as_ptr().add(i));
        let part = vfmaq_f32(vdupq_n_f32(0.0), xv, wv);
        sum += vaddvq_f32(part);
        i += 4;
    }
    while i < len {
        sum += *x.get_unchecked(i) * *w.get_unchecked(i);
        i += 1;
    }
    sum
}

#[inline(always)]
#[cfg(any(target_arch = "aarch64"))]
unsafe fn prefetch_read_l1(ptr: *const f32, bytes_ahead: isize) {
    // Best-effort prefetch: ignore if inline asm not supported on some toolchains.
    // AArch64 prfm pldl1keep (load to L1, keep) hint.
    #[cfg(any(target_arch = "aarch64"))]
    {
        #[allow(unused_unsafe)]
        core::arch::asm!(
            "prfm pldl1keep, [{addr}, #{ofs}]",
            addr = in(reg) ptr,
            ofs = const 0,
            options(nostack, preserves_flags, readonly)
        );
        let _ = bytes_ahead; // not used in simple form
    }
}

#[inline(always)]
pub unsafe fn matmul_rows_neon_contig(
    input: &[f32],
    n: usize,
    c: usize,
    coef_row_major: &[f32],
    e: usize,
    out: &mut [f32],
) {
    debug_assert!(input.len() >= n * c);
    debug_assert!(coef_row_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    for i in 0..n {
        let row_ptr = input.as_ptr().add(i * c);
        let out_ptr = out.as_mut_ptr().add(i * e);

        // Compute 4 output columns at a time to reuse row loads
        let mut j = 0usize;
        while j + 4 <= e {
            let w0_ptr = coef_row_major.as_ptr().add((j + 0) * c);
            let w1_ptr = coef_row_major.as_ptr().add((j + 1) * c);
            let w2_ptr = coef_row_major.as_ptr().add((j + 2) * c);
            let w3_ptr = coef_row_major.as_ptr().add((j + 3) * c);

            // Vector accumulators
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut k = 0usize;
            while k + 16 <= c {
                let x0 = vld1q_f32(row_ptr.add(k + 0));
                let x1 = vld1q_f32(row_ptr.add(k + 4));
                let x2 = vld1q_f32(row_ptr.add(k + 8));
                let x3 = vld1q_f32(row_ptr.add(k + 12));

                let w00 = vld1q_f32(w0_ptr.add(k + 0));
                let w01 = vld1q_f32(w0_ptr.add(k + 4));
                let w02 = vld1q_f32(w0_ptr.add(k + 8));
                let w03 = vld1q_f32(w0_ptr.add(k + 12));
                acc0 = vfmaq_f32(acc0, x0, w00);
                acc0 = vfmaq_f32(acc0, x1, w01);
                acc0 = vfmaq_f32(acc0, x2, w02);
                acc0 = vfmaq_f32(acc0, x3, w03);

                let w10 = vld1q_f32(w1_ptr.add(k + 0));
                let w11 = vld1q_f32(w1_ptr.add(k + 4));
                let w12 = vld1q_f32(w1_ptr.add(k + 8));
                let w13 = vld1q_f32(w1_ptr.add(k + 12));
                acc1 = vfmaq_f32(acc1, x0, w10);
                acc1 = vfmaq_f32(acc1, x1, w11);
                acc1 = vfmaq_f32(acc1, x2, w12);
                acc1 = vfmaq_f32(acc1, x3, w13);

                let w20 = vld1q_f32(w2_ptr.add(k + 0));
                let w21 = vld1q_f32(w2_ptr.add(k + 4));
                let w22 = vld1q_f32(w2_ptr.add(k + 8));
                let w23 = vld1q_f32(w2_ptr.add(k + 12));
                acc2 = vfmaq_f32(acc2, x0, w20);
                acc2 = vfmaq_f32(acc2, x1, w21);
                acc2 = vfmaq_f32(acc2, x2, w22);
                acc2 = vfmaq_f32(acc2, x3, w23);

                let w30 = vld1q_f32(w3_ptr.add(k + 0));
                let w31 = vld1q_f32(w3_ptr.add(k + 4));
                let w32 = vld1q_f32(w3_ptr.add(k + 8));
                let w33 = vld1q_f32(w3_ptr.add(k + 12));
                acc3 = vfmaq_f32(acc3, x0, w30);
                acc3 = vfmaq_f32(acc3, x1, w31);
                acc3 = vfmaq_f32(acc3, x2, w32);
                acc3 = vfmaq_f32(acc3, x3, w33);

                k += 16;
            }

            while k + 4 <= c {
                let x = vld1q_f32(row_ptr.add(k));
                let w0 = vld1q_f32(w0_ptr.add(k));
                let w1 = vld1q_f32(w1_ptr.add(k));
                let w2 = vld1q_f32(w2_ptr.add(k));
                let w3 = vld1q_f32(w3_ptr.add(k));
                acc0 = vfmaq_f32(acc0, x, w0);
                acc1 = vfmaq_f32(acc1, x, w1);
                acc2 = vfmaq_f32(acc2, x, w2);
                acc3 = vfmaq_f32(acc3, x, w3);
                k += 4;
            }

            // Scalar tail for c % 4
            let mut s0 = vaddvq_f32(acc0);
            let mut s1 = vaddvq_f32(acc1);
            let mut s2 = vaddvq_f32(acc2);
            let mut s3 = vaddvq_f32(acc3);
            while k < c {
                let xv = *row_ptr.add(k);
                s0 += xv * *w0_ptr.add(k);
                s1 += xv * *w1_ptr.add(k);
                s2 += xv * *w2_ptr.add(k);
                s3 += xv * *w3_ptr.add(k);
                k += 1;
            }

            *out_ptr.add(j + 0) = s0;
            *out_ptr.add(j + 1) = s1;
            *out_ptr.add(j + 2) = s2;
            *out_ptr.add(j + 3) = s3;

            j += 4;
        }

        // Tail: 1..=3 remaining columns
        while j < e {
            let w_ptr = coef_row_major.as_ptr().add(j * c);
            let mut accv = vdupq_n_f32(0.0);
            let mut k = 0usize;
            while k + 16 <= c {
                let x0 = vld1q_f32(row_ptr.add(k + 0));
                let w0 = vld1q_f32(w_ptr.add(k + 0));
                let x1 = vld1q_f32(row_ptr.add(k + 4));
                let w1 = vld1q_f32(w_ptr.add(k + 4));
                let x2 = vld1q_f32(row_ptr.add(k + 8));
                let w2 = vld1q_f32(w_ptr.add(k + 8));
                let x3 = vld1q_f32(row_ptr.add(k + 12));
                let w3 = vld1q_f32(w_ptr.add(k + 12));
                accv = vfmaq_f32(accv, x0, w0);
                accv = vfmaq_f32(accv, x1, w1);
                accv = vfmaq_f32(accv, x2, w2);
                accv = vfmaq_f32(accv, x3, w3);
                k += 16;
            }
            while k + 4 <= c {
                let x = vld1q_f32(row_ptr.add(k));
                let w = vld1q_f32(w_ptr.add(k));
                accv = vfmaq_f32(accv, x, w);
                k += 4;
            }
            let mut sum = vaddvq_f32(accv);
            while k < c {
                sum += *row_ptr.add(k) * *w_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}

/// Matmul using transposed coefficients (feature-major: [c, e]).
/// Processes output classes in chunks of 4 with NEON and broadcasts input features.
#[inline(always)]
pub unsafe fn matmul_rows_neon_contig_t(
    input: &[f32],
    n: usize,
    c: usize,
    coef_col_major: &[f32], // layout [c, e]
    e: usize,
    out: &mut [f32],
) {
    debug_assert!(input.len() >= n * c);
    debug_assert!(coef_col_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    for i in 0..n {
        let row_ptr = input.as_ptr().add(i * c);
        let out_ptr = out.as_mut_ptr().add(i * e);

        let mut j = 0usize;
        while j + 4 <= e {
            let mut acc = vdupq_n_f32(0.0);

            let mut k = 0usize;
            while k + 4 <= c {
                // Prefetch upcoming inputs and weights lightly
                prefetch_read_l1(row_ptr.add(k + 32), 128);
                let x = vld1q_f32(row_ptr.add(k));

                let w0 = vld1q_f32(coef_col_major.as_ptr().add((k + 0) * e + j));
                let w1 = vld1q_f32(coef_col_major.as_ptr().add((k + 1) * e + j));
                let w2 = vld1q_f32(coef_col_major.as_ptr().add((k + 2) * e + j));
                let w3 = vld1q_f32(coef_col_major.as_ptr().add((k + 3) * e + j));

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
                let xv = vdupq_n_f32(*row_ptr.add(k));
                let wv = vld1q_f32(coef_col_major.as_ptr().add(k * e + j));
                acc = vfmaq_f32(acc, xv, wv);
                k += 1;
            }
            vst1q_f32(out_ptr.add(j), acc);
            j += 4;
        }

        // Tail classes 1..=3: scalar accumulation per class with feature loop
        while j < e {
            let mut sum = 0f32;
            let mut k = 0usize;
            while k + 4 <= c {
                let x = vld1q_f32(row_ptr.add(k));
                let w = vld1q_f32(coef_col_major.as_ptr().add(k * e + j));
                let part = vmulq_f32(x, w);
                sum += vaddvq_f32(part);
                k += 4;
            }
            while k < c {
                sum += *row_ptr.add(k) * *coef_col_major.as_ptr().add(k * e + j);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}

/// Matmul using transposed coefficients with a gathered (strided) input row into rowbuf.
#[inline(always)]
pub unsafe fn matmul_rows_neon_gather_t(
    input_ptr: *const f32,
    n: usize,
    c: usize,
    s0: isize,
    s1: isize,
    coef_col_major: &[f32], // layout [c, e]
    e: usize,
    out: &mut [f32],
    rowbuf: &mut [f32],
) {
    debug_assert!(rowbuf.len() >= c);
    debug_assert!(coef_col_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    if s1 == 1 && s0 == c as isize {
        let input_slice = std::slice::from_raw_parts(input_ptr, n * c);
        return matmul_rows_neon_contig_t(input_slice, n, c, coef_col_major, e, out);
    }

    for i in 0..n {
        // Gather row i into contiguous buffer
        let mut k = 0usize;
        while k + 8 <= c {
            let base = i as isize * s0 + k as isize * s1;
            rowbuf[k + 0] = *input_ptr.offset(base + 0 * s1);
            rowbuf[k + 1] = *input_ptr.offset(base + 1 * s1);
            rowbuf[k + 2] = *input_ptr.offset(base + 2 * s1);
            rowbuf[k + 3] = *input_ptr.offset(base + 3 * s1);
            rowbuf[k + 4] = *input_ptr.offset(base + 4 * s1);
            rowbuf[k + 5] = *input_ptr.offset(base + 5 * s1);
            rowbuf[k + 6] = *input_ptr.offset(base + 6 * s1);
            rowbuf[k + 7] = *input_ptr.offset(base + 7 * s1);
            k += 8;
        }
        while k < c {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
            k += 1;
        }

        // Compute using contig_t on the row buffer
        matmul_rows_neon_contig_t(&rowbuf[..c], 1, c, coef_col_major, e, &mut out[i * e..]);
    }
}

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
    debug_assert!(coef_row_major.len() >= e * c);
    debug_assert!(out.len() >= n * e);

    // Fast path: contiguous rows/cols match a standard row-major 2D array.
    // Detect s1 == 1 (unit stride between elements) and s0 == c (next row).
    if s1 == 1 && s0 == c as isize {
        // Safety: caller guarantees input_ptr points to at least n*c elements
        let input_slice = std::slice::from_raw_parts(input_ptr, n * c);
        return matmul_rows_neon_contig(input_slice, n, c, coef_row_major, e, out);
    }
    for i in 0..n {
        // Gather into a contiguous temporary row buffer
        let mut k = 0usize;
        // Manually unrolled gather to reduce loop overhead
        while k + 8 <= c {
            let base = i as isize * s0 + k as isize * s1;
            rowbuf[k + 0] = *input_ptr.offset(base + 0 * s1);
            rowbuf[k + 1] = *input_ptr.offset(base + 1 * s1);
            rowbuf[k + 2] = *input_ptr.offset(base + 2 * s1);
            rowbuf[k + 3] = *input_ptr.offset(base + 3 * s1);
            rowbuf[k + 4] = *input_ptr.offset(base + 4 * s1);
            rowbuf[k + 5] = *input_ptr.offset(base + 5 * s1);
            rowbuf[k + 6] = *input_ptr.offset(base + 6 * s1);
            rowbuf[k + 7] = *input_ptr.offset(base + 7 * s1);
            k += 8;
        }
        while k < c {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
            k += 1;
        }

        let row_ptr = rowbuf.as_ptr();
        let out_ptr = out.as_mut_ptr().add(i * e);

        // Process 4 columns at a time, same as in contig variant
        let mut j = 0usize;
        while j + 4 <= e {
            let w0_ptr = coef_row_major.as_ptr().add((j + 0) * c);
            let w1_ptr = coef_row_major.as_ptr().add((j + 1) * c);
            let w2_ptr = coef_row_major.as_ptr().add((j + 2) * c);
            let w3_ptr = coef_row_major.as_ptr().add((j + 3) * c);

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut k = 0usize;
            while k + 16 <= c {
                let x0 = vld1q_f32(row_ptr.add(k + 0));
                let x1 = vld1q_f32(row_ptr.add(k + 4));
                let x2 = vld1q_f32(row_ptr.add(k + 8));
                let x3 = vld1q_f32(row_ptr.add(k + 12));

                let w00 = vld1q_f32(w0_ptr.add(k + 0));
                let w01 = vld1q_f32(w0_ptr.add(k + 4));
                let w02 = vld1q_f32(w0_ptr.add(k + 8));
                let w03 = vld1q_f32(w0_ptr.add(k + 12));
                acc0 = vfmaq_f32(acc0, x0, w00);
                acc0 = vfmaq_f32(acc0, x1, w01);
                acc0 = vfmaq_f32(acc0, x2, w02);
                acc0 = vfmaq_f32(acc0, x3, w03);

                let w10 = vld1q_f32(w1_ptr.add(k + 0));
                let w11 = vld1q_f32(w1_ptr.add(k + 4));
                let w12 = vld1q_f32(w1_ptr.add(k + 8));
                let w13 = vld1q_f32(w1_ptr.add(k + 12));
                acc1 = vfmaq_f32(acc1, x0, w10);
                acc1 = vfmaq_f32(acc1, x1, w11);
                acc1 = vfmaq_f32(acc1, x2, w12);
                acc1 = vfmaq_f32(acc1, x3, w13);

                let w20 = vld1q_f32(w2_ptr.add(k + 0));
                let w21 = vld1q_f32(w2_ptr.add(k + 4));
                let w22 = vld1q_f32(w2_ptr.add(k + 8));
                let w23 = vld1q_f32(w2_ptr.add(k + 12));
                acc2 = vfmaq_f32(acc2, x0, w20);
                acc2 = vfmaq_f32(acc2, x1, w21);
                acc2 = vfmaq_f32(acc2, x2, w22);
                acc2 = vfmaq_f32(acc2, x3, w23);

                let w30 = vld1q_f32(w3_ptr.add(k + 0));
                let w31 = vld1q_f32(w3_ptr.add(k + 4));
                let w32 = vld1q_f32(w3_ptr.add(k + 8));
                let w33 = vld1q_f32(w3_ptr.add(k + 12));
                acc3 = vfmaq_f32(acc3, x0, w30);
                acc3 = vfmaq_f32(acc3, x1, w31);
                acc3 = vfmaq_f32(acc3, x2, w32);
                acc3 = vfmaq_f32(acc3, x3, w33);

                k += 16;
            }
            while k + 4 <= c {
                let x = vld1q_f32(row_ptr.add(k));
                let w0 = vld1q_f32(w0_ptr.add(k));
                let w1 = vld1q_f32(w1_ptr.add(k));
                let w2 = vld1q_f32(w2_ptr.add(k));
                let w3 = vld1q_f32(w3_ptr.add(k));
                acc0 = vfmaq_f32(acc0, x, w0);
                acc1 = vfmaq_f32(acc1, x, w1);
                acc2 = vfmaq_f32(acc2, x, w2);
                acc3 = vfmaq_f32(acc3, x, w3);
                k += 4;
            }
            let mut s0 = vaddvq_f32(acc0);
            let mut s1 = vaddvq_f32(acc1);
            let mut s2 = vaddvq_f32(acc2);
            let mut s3 = vaddvq_f32(acc3);
            while k < c {
                let xv = *row_ptr.add(k);
                s0 += xv * *w0_ptr.add(k);
                s1 += xv * *w1_ptr.add(k);
                s2 += xv * *w2_ptr.add(k);
                s3 += xv * *w3_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j + 0) = s0;
            *out_ptr.add(j + 1) = s1;
            *out_ptr.add(j + 2) = s2;
            *out_ptr.add(j + 3) = s3;
            j += 4;
        }

        while j < e {
            let w_ptr = coef_row_major.as_ptr().add(j * c);
            let mut accv = vdupq_n_f32(0.0);
            let mut k = 0usize;
            while k + 16 <= c {
                let x0 = vld1q_f32(row_ptr.add(k + 0));
                let w0 = vld1q_f32(w_ptr.add(k + 0));
                let x1 = vld1q_f32(row_ptr.add(k + 4));
                let w1 = vld1q_f32(w_ptr.add(k + 4));
                let x2 = vld1q_f32(row_ptr.add(k + 8));
                let w2 = vld1q_f32(w_ptr.add(k + 8));
                let x3 = vld1q_f32(row_ptr.add(k + 12));
                let w3 = vld1q_f32(w_ptr.add(k + 12));
                accv = vfmaq_f32(accv, x0, w0);
                accv = vfmaq_f32(accv, x1, w1);
                accv = vfmaq_f32(accv, x2, w2);
                accv = vfmaq_f32(accv, x3, w3);
                k += 16;
            }
            while k + 4 <= c {
                let x = vld1q_f32(row_ptr.add(k));
                let w = vld1q_f32(w_ptr.add(k));
                accv = vfmaq_f32(accv, x, w);
                k += 4;
            }
            let mut sum = vaddvq_f32(accv);
            while k < c {
                sum += *row_ptr.add(k) * *w_ptr.add(k);
                k += 1;
            }
            *out_ptr.add(j) = sum;
            j += 1;
        }
    }
}
