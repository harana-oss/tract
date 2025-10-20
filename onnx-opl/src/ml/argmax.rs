#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
#[cfg(target_arch = "aarch64")]
unsafe fn argmax_row_impl(row: &[f32]) -> usize {
    let len = row.len();
    let ptr = row.as_ptr();
    let mut i = 0usize;

    let mut best_val = core::f32::NEG_INFINITY;
    let mut best_idx = 0usize;

    // 16-wide unrolled with interleaved loads
    while i + 16 <= len {
        // Load 4x4 with single instruction structure
        let v = vld1q_f32_x4(ptr.add(i));

        // Parallel max across all 4 vectors
        let m01 = vmaxq_f32(v.0, v.1);
        let m23 = vmaxq_f32(v.2, v.3);
        let m = vmaxq_f32(m01, m23);
        let chunk_max = vmaxvq_f32(m);

        if chunk_max > best_val {
            // Compare against broadcasted chunk_max
            let cmv = vdupq_n_f32(chunk_max);

            let mask0 = vceqq_f32(v.0, cmv);
            let mask1 = vceqq_f32(v.1, cmv);
            let mask2 = vceqq_f32(v.2, cmv);
            let mask3 = vceqq_f32(v.3, cmv);

            // Narrow comparison results into a compact bitmask (4 bits per lane)
            let narrow01 = vshrn_n_u32::<4>(mask0);
            let narrow23 = vshrn_n_u32::<4>(mask1);
            let narrow45 = vshrn_n_u32::<4>(mask2);
            let narrow67 = vshrn_n_u32::<4>(mask3);

            let combined01 = vcombine_u16(narrow01, narrow23);
            let combined23 = vcombine_u16(narrow45, narrow67);

            let final_narrow1 = vshrn_n_u16::<4>(combined01);
            let final_narrow2 = vshrn_n_u16::<4>(combined23);

            let final_combined = vcombine_u8(final_narrow1, final_narrow2);
            // Extract low 64 bits holding the first eight nibbles (covers all 16 lanes)
            let bitmask = vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(final_combined)), 0);

            if bitmask != 0 {
                // Use reverse_bits + leading_zeros to find index of first set nibble
                let reversed = bitmask.reverse_bits();
                let leading_zeros = reversed.leading_zeros();
                let local = (leading_zeros as usize) >> 2; // 4 bits per lane

                best_val = chunk_max;
                best_idx = i + local;
            }
        }
        i += 16;
    }

    // 8-wide pass
    if i + 8 <= len {
        let v0 = vld1q_f32(ptr.add(i));
        let v1 = vld1q_f32(ptr.add(i + 4));
        let m = vmaxq_f32(v0, v1);
        let chunk_max = vmaxvq_f32(m);

        if chunk_max > best_val {
            let cmv = vdupq_n_f32(chunk_max);
            let mask0 = vceqq_f32(v0, cmv);
            let mask1 = vceqq_f32(v1, cmv);

            let narrow0 = vshrn_n_u32::<4>(mask0);
            let narrow1 = vshrn_n_u32::<4>(mask1);
            let combined = vcombine_u16(narrow0, narrow1);
            let final_narrow = vshrn_n_u16::<4>(combined);

            let bitmask = vget_lane_u64(vreinterpret_u64_u8(final_narrow), 0);

            if bitmask != 0 {
                let local = (bitmask.reverse_bits().leading_zeros() as usize) >> 2;
                best_val = chunk_max;
                best_idx = i + local;
            }
        }
        i += 8;
    }

    // 4-wide pass
    while i + 4 <= len {
        let v = vld1q_f32(ptr.add(i));
        let v_max = vmaxvq_f32(v);

        if v_max > best_val {
            let cmv = vdupq_n_f32(v_max);
            let mask = vceqq_f32(v, cmv);

            let narrow = vshrn_n_u32::<4>(mask);
            let bitmask = vget_lane_u32(vreinterpret_u32_u16(narrow), 0);

            if bitmask != 0 {
                let local = (bitmask.reverse_bits().leading_zeros() as usize) >> 3; // 8 bits per lane after shrn
                best_val = v_max;
                best_idx = i + local;
            }
        }
        i += 4;
    }

    // Scalar tail
    while i < len {
        let v = *ptr.add(i);
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i += 1;
    }

    best_idx
}

#[cfg(target_arch = "aarch64")]
pub fn argmax_row(row: &[f32]) -> usize {
    unsafe { argmax_row_impl(row) }
}

use super::smallvec::SmallVec;
thread_local! {
    static ARGMAX_ROWBUF_TL: std::cell::RefCell<SmallVec<f32, 256>> =
        std::cell::RefCell::new(SmallVec::new());
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn argmax_rows(matrix: &[f32], n: usize, e: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &matrix[i * e..(i + 1) * e];
        out.push(argmax_row(row));
    }
    out
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub unsafe fn argmax_rows_gather_into(
    input_ptr: *const f32,
    n: usize,
    e: usize,
    s0: isize,
    s1: isize,
    rowbuf: &mut [f32],
    out_idx: &mut [usize],
) {
    for i in 0..n {
        for k in 0..e {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
        }
        let row = &rowbuf[..e];
        out_idx[i] = argmax_row(row);
    }
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub unsafe fn argmax_rows_gather(
    input_ptr: *const f32,
    n: usize,
    e: usize,
    s0: isize,
    s1: isize,
) -> Vec<usize> {
    let mut out = vec![0usize; n];
    ARGMAX_ROWBUF_TL.with(|rb| {
        let mut v = rb.borrow_mut();
        let mut h = v.handle();
        if h.len() < e {
            h.resize(e, 0.0);
        }
        argmax_rows_gather_into(input_ptr, n, e, s0, s1, &mut h[..e], &mut out);
    });
    out
}

// ==========================
// x86_64 AVX2 implementation
// ==========================

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn argmax_row_scalar(row: &[f32]) -> usize {
    let mut best = core::f32::NEG_INFINITY;
    let mut idx = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hmax_ps128(v: __m128) -> f32 {
    // Reduce 4-lane vector to max scalar
    let mut m = v;
    let shuf = _mm_movehdup_ps(m); // [m1,m1,m3,m3]
    m = _mm_max_ps(m, shuf);
    let shuf2 = _mm_movehl_ps(shuf, m); // [m3,*,*,*]
    m = _mm_max_ss(m, shuf2);
    _mm_cvtss_f32(m)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hmax_ps256(v: __m256) -> f32 {
    // Max across 8 lanes: reduce to two 128s then horizontally reduce
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(hi, lo);
    hmax_ps128(m128)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hmax_ps512(v: __m512) -> f32 {
    // Reduce 16 lanes: split into two 256-bit halves then reuse hmax_ps256
    let hi = _mm512_extractf32x8_ps(v, 1);
    let lo = _mm512_castps512_ps256(v);
    let m256 = _mm256_max_ps(hi, lo);
    hmax_ps256(m256)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn argmax_row_impl_avx512(row: &[f32]) -> usize {
    use std::arch::x86_64::*;
    let len = row.len();
    let ptr = row.as_ptr();
    let mut i = 0usize;

    let mut best_val = core::f32::NEG_INFINITY;
    let mut best_idx = 0usize;

    // 64-wide (4x __m512) block
    while i + 64 <= len {
        let v0 = _mm512_loadu_ps(ptr.add(i + 0));
        let v1 = _mm512_loadu_ps(ptr.add(i + 16));
        let v2 = _mm512_loadu_ps(ptr.add(i + 32));
        let v3 = _mm512_loadu_ps(ptr.add(i + 48));

        let m01 = _mm512_max_ps(v0, v1);
        let m23 = _mm512_max_ps(v2, v3);
        let m = _mm512_max_ps(m01, m23);
        let chunk_max = hmax_ps512(m);

        if chunk_max > best_val {
            let cmv = _mm512_set1_ps(chunk_max);

            let mask0 = _mm512_cmp_ps_mask(v0, cmv, _CMP_EQ_OQ) as u32;
            if mask0 != 0 {
                let local = mask0.trailing_zeros() as usize; // first equal in v0
                best_val = chunk_max;
                best_idx = i + local;
            } else {
                let mask1 = _mm512_cmp_ps_mask(v1, cmv, _CMP_EQ_OQ) as u32;
                if mask1 != 0 {
                    let local = mask1.trailing_zeros() as usize;
                    best_val = chunk_max;
                    best_idx = i + 16 + local;
                } else {
                    let mask2 = _mm512_cmp_ps_mask(v2, cmv, _CMP_EQ_OQ) as u32;
                    if mask2 != 0 {
                        let local = mask2.trailing_zeros() as usize;
                        best_val = chunk_max;
                        best_idx = i + 32 + local;
                    } else {
                        let mask3 = _mm512_cmp_ps_mask(v3, cmv, _CMP_EQ_OQ) as u32;
                        if mask3 != 0 {
                            let local = mask3.trailing_zeros() as usize;
                            best_val = chunk_max;
                            best_idx = i + 48 + local;
                        }
                    }
                }
            }
        }
        i += 64;
    }

    // 32-wide via two 512 halves (use 256 ops under AVX-512)
    if i + 32 <= len {
        let v0 = _mm512_loadu_ps(ptr.add(i));
        let v1 = _mm512_loadu_ps(ptr.add(i + 16));
        let m = _mm512_max_ps(v0, v1);
        let chunk_max = hmax_ps512(m);
        if chunk_max > best_val {
            let cmv = _mm512_set1_ps(chunk_max);
            let mask0 = _mm512_cmp_ps_mask(v0, cmv, _CMP_EQ_OQ) as u32;
            if mask0 != 0 {
                let local = mask0.trailing_zeros() as usize;
                best_val = chunk_max;
                best_idx = i + local;
            } else {
                let mask1 = _mm512_cmp_ps_mask(v1, cmv, _CMP_EQ_OQ) as u32;
                if mask1 != 0 {
                    let local = mask1.trailing_zeros() as usize;
                    best_val = chunk_max;
                    best_idx = i + 16 + local;
                }
            }
        }
        i += 32;
    }

    // 16-wide
    if i + 16 <= len {
        let v = _mm512_loadu_ps(ptr.add(i));
        let v_max = hmax_ps512(v);
        if v_max > best_val {
            let cmv = _mm512_set1_ps(v_max);
            let mask = _mm512_cmp_ps_mask(v, cmv, _CMP_EQ_OQ) as u32;
            if mask != 0 {
                let local = mask.trailing_zeros() as usize;
                best_val = v_max;
                best_idx = i + local;
            }
        }
        i += 16;
    }

    // 8-wide tail: reuse AVX2 path if available in this function
    if i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        let v_max = hmax_ps256(v);
        if v_max > best_val {
            let cmv = _mm256_set1_ps(v_max);
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(v, cmv, _CMP_EQ_OQ)) as u32;
            if mask != 0 {
                let local = mask.trailing_zeros() as usize;
                best_val = v_max;
                best_idx = i + local;
            }
        }
        i += 8;
    }

    while i < len {
        let v = *ptr.add(i);
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i += 1;
    }

    best_idx
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_row_impl_avx2(row: &[f32]) -> usize {
    let len = row.len();
    let ptr = row.as_ptr();
    let mut i = 0usize;

    let mut best_val = core::f32::NEG_INFINITY;
    let mut best_idx = 0usize;

    // 32-wide (4x __m256) block
    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(ptr.add(i + 0));
        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
        let v3 = _mm256_loadu_ps(ptr.add(i + 24));

        let m01 = _mm256_max_ps(v0, v1);
        let m23 = _mm256_max_ps(v2, v3);
        let m = _mm256_max_ps(m01, m23);
        let chunk_max = hmax_ps256(m);

        if chunk_max > best_val {
            let cmv = _mm256_set1_ps(chunk_max);
            let mask0 = _mm256_movemask_ps(_mm256_cmp_ps(v0, cmv, _CMP_EQ_OQ)) as u32;
            if mask0 != 0 {
                let local = mask0.trailing_zeros() as usize; // first set bit
                best_val = chunk_max;
                best_idx = i + local;
            } else {
                let mask1 = _mm256_movemask_ps(_mm256_cmp_ps(v1, cmv, _CMP_EQ_OQ)) as u32;
                if mask1 != 0 {
                    let local = mask1.trailing_zeros() as usize;
                    best_val = chunk_max;
                    best_idx = i + 8 + local;
                } else {
                    let mask2 = _mm256_movemask_ps(_mm256_cmp_ps(v2, cmv, _CMP_EQ_OQ)) as u32;
                    if mask2 != 0 {
                        let local = mask2.trailing_zeros() as usize;
                        best_val = chunk_max;
                        best_idx = i + 16 + local;
                    } else {
                        let mask3 = _mm256_movemask_ps(_mm256_cmp_ps(v3, cmv, _CMP_EQ_OQ)) as u32;
                        if mask3 != 0 {
                            let local = mask3.trailing_zeros() as usize;
                            best_val = chunk_max;
                            best_idx = i + 24 + local;
                        }
                    }
                }
            }
        }
        i += 32;
    }

    // 16-wide
    if i + 16 <= len {
        let v0 = _mm256_loadu_ps(ptr.add(i + 0));
        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
        let m = _mm256_max_ps(v0, v1);
        let chunk_max = hmax_ps256(m);

        if chunk_max > best_val {
            let cmv = _mm256_set1_ps(chunk_max);
            let mask0 = _mm256_movemask_ps(_mm256_cmp_ps(v0, cmv, _CMP_EQ_OQ)) as u32;
            if mask0 != 0 {
                let local = mask0.trailing_zeros() as usize;
                best_val = chunk_max;
                best_idx = i + local;
            } else {
                let mask1 = _mm256_movemask_ps(_mm256_cmp_ps(v1, cmv, _CMP_EQ_OQ)) as u32;
                if mask1 != 0 {
                    let local = mask1.trailing_zeros() as usize;
                    best_val = chunk_max;
                    best_idx = i + 8 + local;
                }
            }
        }
        i += 16;
    }

    // 8-wide
    if i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        let v_max = hmax_ps256(v);
        if v_max > best_val {
            let cmv = _mm256_set1_ps(v_max);
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(v, cmv, _CMP_EQ_OQ)) as u32;
            if mask != 0 {
                let local = mask.trailing_zeros() as usize;
                best_val = v_max;
                best_idx = i + local;
            }
        }
        i += 8;
    }

    // Scalar tail
    while i < len {
        let v = *ptr.add(i);
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i += 1;
    }

    best_idx
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn argmax_row(row: &[f32]) -> usize {
    // Prefer AVX-512 when available; else AVX2; else scalar
    if std::is_x86_feature_detected!("avx512f") {
        log::debug!("argmax: using AVX512 path");
        unsafe { argmax_row_impl_avx512(row) }
    } else if std::is_x86_feature_detected!("avx2") {
        log::debug!("argmax: using AVX2 path");
        unsafe { argmax_row_impl_avx2(row) }
    } else {
        argmax_row_scalar(row)
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn argmax_rows(matrix: &[f32], n: usize, e: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &matrix[i * e..(i + 1) * e];
        out.push(argmax_row(row));
    }
    out
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn argmax_rows_gather_into(
    input_ptr: *const f32,
    n: usize,
    e: usize,
    s0: isize,
    s1: isize,
    rowbuf: &mut [f32],
    out_idx: &mut [usize],
) {
    for i in 0..n {
        for k in 0..e {
            let off = i as isize * s0 + k as isize * s1;
            rowbuf[k] = *input_ptr.offset(off);
        }
        let row = &rowbuf[..e];
        out_idx[i] = argmax_row(row);
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn argmax_rows_gather(
    input_ptr: *const f32,
    n: usize,
    e: usize,
    s0: isize,
    s1: isize,
) -> Vec<usize> {
    let mut out = vec![0usize; n];
    ARGMAX_ROWBUF_TL.with(|rb| {
        let mut v = rb.borrow_mut();
        let mut h = v.handle();
        if h.len() < e {
            h.resize(e, 0.0);
        }
        argmax_rows_gather_into(input_ptr, n, e, s0, s1, &mut h[..e], &mut out);
    });
    out
}
