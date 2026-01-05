#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-accelerated argmax over a single row, preserving first-maximum semantics.
#[cfg(target_arch = "aarch64")]
unsafe fn argmax_row_neon(row: &[f32]) -> usize {
    let len = row.len();
    let ptr = row.as_ptr();
    let mut i = 0usize;

    // Track best value and index as scalars to preserve first-occurrence ordering across chunks.
    let mut best_val = core::f32::NEG_INFINITY;
    let mut best_idx = 0usize;

    // 16-wide unrolled scan
    while i + 16 <= len {
        let v0 = unsafe { vld1q_f32(ptr.add(i)) };
        let v1 = unsafe { vld1q_f32(ptr.add(i + 4)) };
        let v2 = unsafe { vld1q_f32(ptr.add(i + 8)) };
        let v3 = unsafe { vld1q_f32(ptr.add(i + 12)) };
        // Reduce each block to a scalar max quickly
        let m01 = vmaxq_f32(v0, v1);
        let m23 = vmaxq_f32(v2, v3);
        let m = vmaxq_f32(m01, m23);
        let chunk_max = vmaxvq_f32(m);
        if chunk_max > best_val {
            // Find first index of chunk_max within the 16-lane chunk
            // Check v0..v3 in order and lanes in increasing order to preserve first occurrence
            // vceqq produces mask lanes; we then test per-lane scalars
            let lanes = [v0, v1, v2, v3];
            let mut local = 0usize;
            'outer: for (b, vv) in lanes.iter().enumerate() {
                let mask = vceqq_f32(*vv, vdupq_n_f32(chunk_max));
                // Extract lanes as u32 mask values (all ones for true)
                let arr: [u32; 4] = unsafe { core::mem::transmute(mask) };
                for l in 0..4 {
                    if arr[l] == u32::MAX {
                        local = b * 4 + l;
                        break 'outer;
                    }
                }
            }
            best_val = chunk_max;
            best_idx = i + local;
        }
        i += 16;
    }

    // Handle remaining 4-lane chunks
    while i + 4 <= len {
        let v = unsafe { vld1q_f32(ptr.add(i)) };
        let v_max = vmaxvq_f32(v);
        if v_max > best_val {
            let mask = vceqq_f32(v, vdupq_n_f32(v_max));
            let arr: [u32; 4] = unsafe { core::mem::transmute(mask) };
            for l in 0..4 {
                if arr[l] == u32::MAX {
                    best_val = v_max;
                    best_idx = i + l;
                    break;
                }
            }
        }
        i += 4;
    }

    // Tail scalar scan (also covers entirely short rows)
    while i < len {
        let v = unsafe { *ptr.add(i) };
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i += 1;
    }
    best_idx
}

use super::smallvec::SmallVec;
thread_local! {
    // Inline capacity tuned for common small row sizes; grows to heap if needed
    static ARGMAX_ROWBUF_TL: std::cell::RefCell<SmallVec<f32, 256>> =
        std::cell::RefCell::new(SmallVec::new());
}

/// Compute argmax for each row of a matrix stored row-major in `matrix` with shape (n, e) (contiguous case).
#[inline(always)]
pub fn argmax_rows(matrix: &[f32], n: usize, e: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &matrix[i * e..(i + 1) * e];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            out.push(argmax_row_neon(row));
        }
    }
    out
}

/// Compute argmax for rows gathered from a strided input pointer into a scratch buffer, writing indices into `out_idx`.
/// Strides s0 (row) and s1 (inner) are in elements (not bytes).
#[inline(always)]
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
        #[cfg(target_arch = "aarch64")]
        {
            out_idx[i] = argmax_row_neon(row);
        }
    }
}

/// Convenience wrapper that allocates a thread-local scratch row buffer and returns the indices.
#[inline(always)]
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
