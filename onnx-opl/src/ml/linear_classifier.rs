#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_arch = "aarch64"))]
compile_error!("NEON-only build: linear_classifier requires target_arch = aarch64");

use super::smallvec::SmallVec;
use crate::ml::math;
use crate::ml::{argmax, softmax};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use tract_nnef::internal::*;

thread_local! {
    static ROWBUF_TL: RefCell<SmallVec<f32, 256>> = RefCell::new(SmallVec::new());
    static COMPACT_TL: RefCell<SmallVec<f32, 256>> = RefCell::new(SmallVec::new());
}

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_linear_classifier",
        &parameters(),
        &[("label", TypeName::Scalar.tensor()), ("scores", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

pub fn parse_post_transform(s: &str) -> TractResult<PostTransformLC> {
    match s {
        "NONE" => Ok(PostTransformLC::None),
        "SOFTMAX" => Ok(PostTransformLC::Softmax),
        "LOGISTIC" => Ok(PostTransformLC::Logistic),
        "SOFTMAX_ZERO" | "PROBIT" => bail!("SOFTMAX_ZERO and PROBIT unsupported"),
        _ => bail!("Invalid post_transform: {}", s),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransformLC {
    None,
    Softmax,
    Logistic,
}

#[derive(Debug, Clone)]
pub struct LinearClassifierData {
    pub labels: Arc<Tensor>,
    pub coefficients: Arc<Tensor>,
    pub intercepts: Option<Arc<Tensor>>,
    // Pre-computed metadata / optimized storage
    pub eff_e: usize,  // effective number of weight vectors (1 for compact binary)
    pub feat_c: usize, // feature count
    pub coefficients_raw: Arc<[f32]>, // row-major (eff_e, feat_c)
    pub coefficients_t: Arc<[f32]>, // transposed feature-major (feat_c, eff_e)
    // Pre-evaluated intercepts/flags
    pub binary_compact: bool,                  // eff_e == 1 && classes == 2
    pub bias_scalar: f32,                      // intercepts == Some([b])
    pub intercepts_eff_e: Option<Arc<[f32]>>,  // intercepts per eff_e classes
    pub binary_intercepts_2: Option<[f32; 2]>, // intercepts per class in binary expansion
}

impl Hash for LinearClassifierData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.labels.hash(state);
        self.coefficients.hash(state);
        self.intercepts.hash(state);
        self.eff_e.hash(state);
        self.feat_c.hash(state);
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearClassifier {
    pub data: LinearClassifierData,
    pub multi_class: i64,
    pub post_transform: PostTransformLC,
}

impl LinearClassifier {
    #[inline(always)]
    fn tensor1_from_vec_string(v: Vec<String>) -> TractResult<Tensor> {
        let len = v.len();
        let mut t = unsafe { Tensor::uninitialized::<String>(&[len]) }?;
        t.as_slice_mut::<String>()?.clone_from_slice(&v);
        Ok(t)
    }

    #[inline(always)]
    fn tensor1_from_vec_i64(v: Vec<i64>) -> TractResult<Tensor> {
        let len = v.len();
        let mut t = unsafe { Tensor::uninitialized::<i64>(&[len]) }?;
        t.as_slice_mut::<i64>()?.clone_from_slice(&v);
        Ok(t)
    }
    pub fn n_classes(&self) -> usize {
        self.data.labels.len()
    }
    pub fn labels(&self) -> &Arc<Tensor> {
        &self.data.labels
    }

    // ---- NEON helpers ----
    #[inline(always)]
    fn with_rowbuf<F: FnOnce(&mut [f32]) -> R, R>(len: usize, f: F) -> R {
        ROWBUF_TL.with(|rb| {
            let mut v = rb.borrow_mut();
            let mut h = v.handle();
            if h.len() < len {
                h.resize(len, 0.0);
            }
            f(&mut h[..len])
        })
    }

    #[inline(always)]
    fn with_compact<F: FnOnce(&mut [f32]) -> R, R>(len: usize, f: F) -> R {
        COMPACT_TL.with(|cb| {
            let mut v = cb.borrow_mut();
            let mut h = v.handle();
            if h.len() < len {
                h.resize(len, 0.0);
            }
            f(&mut h[..len])
        })
    }

    #[inline(always)]
    fn logistic_inplace(scores: &mut [f32], n: usize, e: usize) {
        for v in scores.iter_mut().take(n * e) {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    #[inline(always)]
    fn labels_from_argmax(&self, argmax: &[usize]) -> TractResult<Tensor> {
        if self.data.labels.datum_type() == DatumType::String {
            let label_slice = self.data.labels.as_slice::<String>()?;
            let mapped: Vec<String> =
                argmax.iter().map(|&i| label_slice.get(i).cloned().unwrap_or_default()).collect();
            Self::tensor1_from_vec_string(mapped)
        } else if self.data.labels.datum_type() == DatumType::I64 {
            let label_slice = self.data.labels.as_slice::<i64>()?;
            let mapped: Vec<i64> =
                argmax.iter().map(|&i| label_slice.get(i).copied().unwrap_or(0)).collect();
            Self::tensor1_from_vec_i64(mapped)
        } else {
            bail!("Unsupported label type: {:?}", self.data.labels.datum_type())
        }
    }

    // Fused evaluation path producing scores and (optionally) argmax in one pass.
    // Returns (scores_tensor, argmax_indices)
    fn eval_scores_and_argmax_from_tensor(
        &self,
        input: &Tensor,
        want_argmax: bool,
    ) -> TractResult<(Tensor, Option<Vec<usize>>)> {
        let ishape = input.shape();
        let (n, c) = match ishape {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let e = self.n_classes();
        let eff_e = self.data.eff_e;
        let feat_c = self.data.feat_c;
        if feat_c != c {
            bail!("Feature count mismatch: expected {}, got {}", feat_c, c);
        }
        let coef_slice: &[f32] = &self.data.coefficients_raw;
        let intercepts_eff_e: Option<&[f32]> = self.data.intercepts_eff_e.as_deref();
        let bias_scalar = self.data.bias_scalar;
        let binary_intercepts_2 = self.data.binary_intercepts_2;

        // Input data access: prefer contiguous slice, fallback to strided reads
        let input_contig = input.as_slice::<f32>().ok();
        let input_ptr = if input_contig.is_none() { Some(input.as_ptr::<f32>()?) } else { None };
        let strides = input.strides().to_vec();
        let s0 = if n == 1 { 0 } else { strides[0] };
        let s1 = strides[ishape.len() - 1];

        // Fast binary single-vector path when compact (eff_e==1,e==2)
        if eff_e == 1 && e == 2 {
            // For eff_e==1 row-major slice is contiguous weight vector length feat_c
            let w = &coef_slice[..feat_c];
            let bias = bias_scalar;

            // Allocate output tensor directly and fill its slice
            let mut scores_tensor = unsafe { Tensor::uninitialized::<f32>(&[n, 2]) }?;
            let scores = scores_tensor.as_slice_mut::<f32>()?;
            let mut argmax: Option<Vec<usize>> = want_argmax.then(|| Vec::with_capacity(n));
            // scratch buffer for strided rows (thread-local)
            match self.post_transform {
                PostTransformLC::None => {
                    // timing disabled
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        scores[i * 2] = -z;
                        scores[i * 2 + 1] = z;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            scores[i * 2] += b0;
                            scores[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if z >= 0.0 { 1 } else { 0 });
                        }
                    }
                }
                PostTransformLC::Logistic => {
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        let p = 1.0 / (1.0 + (-z).exp());
                        scores[i * 2] = 1.0 - p;
                        scores[i * 2 + 1] = p;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            scores[i * 2] += b0;
                            scores[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if p >= 0.5 { 1 } else { 0 });
                        }
                    }
                }
                PostTransformLC::Softmax => {
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        let t = 2.0 * z;
                        let p = 1.0 / (1.0 + (-t).exp());
                        scores[i * 2] = 1.0 - p;
                        scores[i * 2 + 1] = p;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            scores[i * 2] += b0;
                            scores[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if p >= 0.5 { 1 } else { 0 });
                        }
                    }
                }
            }
            return Ok((scores_tensor, argmax));
        }

        // General path: compute X * W^T using NEON only into a Tensor buffer
        let mut scores_tensor = unsafe { Tensor::uninitialized::<f32>(&[n, eff_e]) }?;
        let scores = scores_tensor.as_slice_mut::<f32>()?;
        unsafe {
            if let Some(x) = &input_contig {
                math::matmul_rows_neon_contig(x, n, c, coef_slice, eff_e, scores);
            } else {
                let base = input_ptr.unwrap();
                Self::with_rowbuf(c, |rowbuf| {
                    math::matmul_rows_neon_gather(
                        base, n, c, s0, s1, coef_slice, eff_e, scores, rowbuf,
                    );
                });
            }
        }

        if let Some(intercepts) = intercepts_eff_e {
            for row in scores.chunks_mut(eff_e) {
                for (v, b) in row.iter_mut().zip(intercepts.iter()) {
                    *v += *b;
                }
            }
        } else if eff_e == 1 && bias_scalar != 0.0 {
            for v in scores.iter_mut() {
                *v += bias_scalar;
            }
        }
        match self.post_transform {
            PostTransformLC::None => {}
            PostTransformLC::Softmax => {
                softmax::softmax_inplace_rows(scores, n, eff_e);
            }
            PostTransformLC::Logistic => {
                Self::logistic_inplace(scores, n, eff_e);
            }
        }
        if eff_e == 1 && e == 2 {
            let compact = scores;
            // Expand into a new Tensor of shape [n,2]
            let mut full_tensor = unsafe { Tensor::uninitialized::<f32>(&[n, 2]) }?;
            let full = full_tensor.as_slice_mut::<f32>()?;
            match self.post_transform {
                PostTransformLC::None => {
                    for i in 0..n {
                        let v = compact[i];
                        full[i * 2] = -v;
                        full[i * 2 + 1] = v;
                    }
                }
                _ => {
                    for i in 0..n {
                        let v = compact[i];
                        full[i * 2] = 1.0 - v;
                        full[i * 2 + 1] = v;
                    }
                }
            }
            if let Some([b0, b1]) = binary_intercepts_2 {
                for row in full.chunks_mut(2) {
                    row[0] += b0;
                    row[1] += b1;
                }
            }

            let mut argmax = want_argmax.then(|| Vec::with_capacity(n));
            if let Some(ref mut a) = argmax {
                for row in full.chunks(2) {
                    a.push(if row[1] >= row[0] { 1 } else { 0 });
                }
            }
            return Ok((full_tensor, argmax));
        }
        let mut argmax = want_argmax.then(|| Vec::with_capacity(n));
        if let Some(ref mut a) = argmax {
            let indices = argmax::argmax_rows(scores, n, eff_e);
            a.extend(indices);
        }
        Ok((scores_tensor, argmax))
    }

    pub fn eval_scores_into(
        &self,
        input: &Tensor,
        out: &mut [f32],
        want_argmax: bool,
    ) -> TractResult<(usize /*n*/, usize /*m_out*/, Option<Vec<usize>>)> {
        let ishape = input.shape();
        let (n, c) = match ishape {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let e = self.n_classes();
        let eff_e = self.data.eff_e;
        let feat_c = self.data.feat_c;
        if feat_c != c {
            bail!("Feature count mismatch: expected {}, got {}", feat_c, c);
        }
        let m_out = if eff_e == 1 && e == 2 { 2 } else { eff_e };
        ensure!(
            out.len() == n * m_out,
            "output buffer has len {}, expected {}",
            out.len(),
            n * m_out
        );

        let coef_slice: &[f32] = &self.data.coefficients_raw;
        let intercepts_eff_e: Option<&[f32]> = self.data.intercepts_eff_e.as_deref();
        let bias_scalar = self.data.bias_scalar;
        let binary_intercepts_2 = self.data.binary_intercepts_2;

        // Input access
        let input_contig = input.as_slice::<f32>().ok();
        let input_ptr = if input_contig.is_none() { Some(input.as_ptr::<f32>()?) } else { None };
        let strides = input.strides().to_vec();
        let s0 = if n == 1 { 0 } else { strides[0] };
        let s1 = strides[ishape.len() - 1];

        // Fast binary single-vector path when compact (eff_e==1,e==2)
        if eff_e == 1 && e == 2 {
            let w = &coef_slice[..feat_c];
            let bias = bias_scalar;
            let mut argmax: Option<Vec<usize>> = want_argmax.then(|| Vec::with_capacity(n));
            match self.post_transform {
                PostTransformLC::None => {
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        out[i * 2] = -z;
                        out[i * 2 + 1] = z;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            out[i * 2] += b0;
                            out[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if z >= 0.0 { 1 } else { 0 });
                        }
                    }
                }
                PostTransformLC::Logistic => {
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        let p = 1.0 / (1.0 + (-z).exp());
                        out[i * 2] = 1.0 - p;
                        out[i * 2 + 1] = p;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            out[i * 2] += b0;
                            out[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if p >= 0.5 { 1 } else { 0 });
                        }
                    }
                }
                PostTransformLC::Softmax => {
                    for i in 0..n {
                        let mut z: f32;
                        unsafe {
                            if let Some(x) = &input_contig {
                                let row = &x[i * c..(i + 1) * c];
                                z = math::dot_neon(row, w);
                            } else {
                                let base = input_ptr.unwrap();
                                z = Self::with_rowbuf(c, |rowbuf| {
                                    for k in 0..c {
                                        let off = i as isize * s0 + k as isize * s1;
                                        rowbuf[k] = *base.offset(off);
                                    }
                                    math::dot_neon(&rowbuf[..c], w)
                                });
                            }
                        }
                        z += bias;
                        let t = 2.0 * z;
                        let p = 1.0 / (1.0 + (-t).exp());
                        out[i * 2] = 1.0 - p;
                        out[i * 2 + 1] = p;
                        if let Some([b0, b1]) = binary_intercepts_2 {
                            out[i * 2] += b0;
                            out[i * 2 + 1] += b1;
                        }
                        if let Some(ref mut a) = argmax {
                            a.push(if p >= 0.5 { 1 } else { 0 });
                        }
                    }
                }
            }
            return Ok((n, m_out, argmax));
        }

        // General path
        if let Some(x) = &input_contig {
            if m_out == eff_e {
                unsafe {
                    math::matmul_rows_neon_contig(x, n, c, coef_slice, eff_e, out);
                }
            } else {
                // compact compute into scratch then expand to out (binary case)
                debug_assert!(eff_e == 1 && e == 2);
                let w = &coef_slice[..c];
                Self::with_compact(n, |compact| {
                    unsafe {
                        for i in 0..n {
                            let row = &x[i * c..(i + 1) * c];
                            compact[i] = math::dot_neon(row, w);
                        }
                    }
                    // intercepts
                    if bias_scalar != 0.0 {
                        for v in compact.iter_mut().take(n) {
                            *v += bias_scalar;
                        }
                    }
                    // post transform
                    match self.post_transform {
                        PostTransformLC::None => {}
                        PostTransformLC::Softmax => {
                            softmax::softmax_inplace_rows(&mut compact[..n], n, 1)
                        }
                        PostTransformLC::Logistic => {
                            Self::logistic_inplace(&mut compact[..n], n, 1)
                        }
                    }
                    // expand to out (and optional per-class intercepts)
                    for i in 0..n {
                        let v = compact[i];
                        match self.post_transform {
                            PostTransformLC::None => {
                                out[i * 2] = -v;
                                out[i * 2 + 1] = v;
                            }
                            _ => {
                                out[i * 2] = 1.0 - v;
                                out[i * 2 + 1] = v;
                            }
                        }
                    }
                    if let Some([b0, b1]) = binary_intercepts_2 {
                        for row in out.chunks_mut(2) {
                            row[0] += b0;
                            row[1] += b1;
                        }
                    }
                });
            }
        } else {
            // strided input
            if m_out == eff_e {
                unsafe {
                    let base = input_ptr.unwrap();
                    Self::with_rowbuf(c, |rowbuf| {
                        math::matmul_rows_neon_gather(
                            base, n, c, s0, s1, coef_slice, eff_e, out, rowbuf,
                        );
                    });
                }
            } else {
                debug_assert!(eff_e == 1 && e == 2);
                let w = &coef_slice[..c];
                Self::with_compact(n, |compact| {
                    let base = input_ptr.unwrap();
                    Self::with_rowbuf(c, |rowbuf| {
                        for i in 0..n {
                            for k in 0..c {
                                let off = i as isize * s0 + k as isize * s1;
                                rowbuf[k] = unsafe { *base.offset(off) };
                            }
                            compact[i] = unsafe { math::dot_neon(&rowbuf[..c], w) };
                        }
                    });
                    if bias_scalar != 0.0 {
                        for v in compact.iter_mut().take(n) {
                            *v += bias_scalar;
                        }
                    }
                    match self.post_transform {
                        PostTransformLC::None => {}
                        PostTransformLC::Softmax => {
                            softmax::softmax_inplace_rows(&mut compact[..n], n, 1)
                        }
                        PostTransformLC::Logistic => {
                            Self::logistic_inplace(&mut compact[..n], n, 1)
                        }
                    }
                    for i in 0..n {
                        let v = compact[i];
                        match self.post_transform {
                            PostTransformLC::None => {
                                out[i * 2] = -v;
                                out[i * 2 + 1] = v;
                            }
                            _ => {
                                out[i * 2] = 1.0 - v;
                                out[i * 2 + 1] = v;
                            }
                        }
                    }
                    if let Some([b0, b1]) = binary_intercepts_2 {
                        for row in out.chunks_mut(2) {
                            row[0] += b0;
                            row[1] += b1;
                        }
                    }
                });
            }
        }

        // Add intercepts (per-class) for m_out == eff_e case
        if m_out == eff_e {
            if let Some(intercepts) = intercepts_eff_e {
                for row in out.chunks_mut(eff_e) {
                    for (v, b) in row.iter_mut().zip(intercepts.iter()) {
                        *v += *b;
                    }
                }
            } else if eff_e == 1 && bias_scalar != 0.0 {
                for v in out.iter_mut() {
                    *v += bias_scalar;
                }
            }
            match self.post_transform {
                PostTransformLC::None => {}
                PostTransformLC::Softmax => softmax::softmax_inplace_rows(out, n, eff_e),
                PostTransformLC::Logistic => Self::logistic_inplace(out, n, eff_e),
            }
        }

        let mut argmax = want_argmax.then(|| Vec::with_capacity(n));
        if let Some(ref mut a) = argmax {
            if m_out == 2 && eff_e == 1 && e == 2 {
                for row in out.chunks(2) {
                    a.push(if row[1] >= row[0] { 1 } else { 0 });
                }
            } else {
                let indices = argmax::argmax_rows(out, n, m_out);
                a.extend(indices);
            }
        }

        Ok((n, m_out, argmax))
    }

    /// Compute scores, then normalize them in-place per row using the requested norm.
    /// This avoids allocating a second buffer and a separate normalization pass.
    pub fn eval_scores_normalized_into(
        &self,
        input: &Tensor,
        out: &mut [f32],
        want_argmax: bool,
        norm_kind: crate::ml::normalizer::NormKind,
    ) -> TractResult<(usize /*n*/, usize /*m_out*/, Option<Vec<usize>>)> {
        // If Softmax is requested and the normalization is L1, skip Normalizer: softmax already
        // produces L1-normalized rows. Avoids an unnecessary extra pass.
        if self.post_transform == PostTransformLC::Softmax
            && matches!(norm_kind, crate::ml::normalizer::NormKind::L1)
        {
            return self.eval_scores_into(input, out, want_argmax);
        }
        let (n, m, argmax) = self.eval_scores_into(input, out, want_argmax)?;
        let normalizer = crate::ml::normalizer::Normalizer { kind: norm_kind };
        normalizer.eval_inplace_rows(out, n, m)?;
        Ok((n, m, argmax))
    }

    pub fn eval_from_tensor(&self, input: &Tensor) -> TractResult<(Tensor, Tensor)> {
        let (scores, argmax) = self.eval_scores_and_argmax_from_tensor(input, true)?;
        let labels = if let Some(argmax) = argmax {
            self.labels_from_argmax(&argmax)?
        } else {
            self.compute_labels(&scores)?
        };
        Ok((labels, scores))
    }

    fn compute_labels(&self, scores: &Tensor) -> TractResult<Tensor> {
        let shape = scores.shape();
        let (n, e) = match shape {
            [n, e] => (*n, *e),
            other => bail!("scores must be 2D, got {} dims", other.len()),
        };
        let data = scores.as_slice::<f32>()?;
        let argmax_indices = argmax::argmax_rows(data, n, e);

        let labels = if self.data.labels.datum_type() == DatumType::String {
            let label_slice = self.data.labels.as_slice::<String>()?;
            let mapped: Vec<String> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).cloned().unwrap_or_default())
                .collect();
            Self::tensor1_from_vec_string(mapped)?
        } else if self.data.labels.datum_type() == DatumType::I64 {
            let label_slice = self.data.labels.as_slice::<i64>()?;
            let mapped: Vec<i64> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).copied().unwrap_or(0))
                .collect();
            Self::tensor1_from_vec_i64(mapped)?
        } else {
            bail!("Unsupported label type: {:?}", self.data.labels.datum_type())
        };

        Ok(labels)
    }
}

impl Op for LinearClassifier {
    fn name(&self) -> StaticName {
        "LinearClassifier".into()
    }

    op_as_typed_op!();
}

impl EvalOp for LinearClassifier {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let (labels, scores) = self.eval_from_tensor(&input)?;
        Ok(tvec!(labels.into_tvalue(), scores.into_tvalue()))
    }
}

impl TypedOp for LinearClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        let label_dt = self.data.labels.datum_type();
        Ok(tvec!(label_dt.fact(&[n.clone()]), f32::fact(&[n.clone(), self.n_classes().into()])))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        // Fuse when LC's scores (outlet 1) are consumed by a single Normalizer.
        // We will replace LC -> Normalizer with a single op that computes LC then normalizes scores.
        if node.outputs.len() < 2 {
            return Ok(None);
        }
        let succs = &model.node(node.id).outputs[1].successors;
        if succs.len() != 1 {
            return Ok(None);
        }
        let succ_inlet = succs[0];
        let succ_node = model.node(succ_inlet.node);
        let Some(norm) = succ_node.op_as::<crate::ml::normalizer::Normalizer>() else {
            return Ok(None);
        };

        // Build a patch that replaces LinearClassifier -> Normalizer(scores) with a single fused op
        let mut patch = TypedModelPatch::default();
        let fused_input = patch.taps(model, &[node.inputs[0]])?[0];
        let fused = PostNormalizedLinearClassifier { lc: self.clone(), norm_kind: norm.kind };
        let out =
            patch.wire_node(format!("{}._fused_lc_postnorm", node.name), fused, &[fused_input])?;
        // Redirect labels users from LC[0] to fused[0]
        patch.shunt_outside(model, OutletId::new(node.id, 0), out[0])?;
        // Redirect scores users from Normalizer[0] to fused[1]
        patch.shunt_outside(model, OutletId::new(succ_node.id, 0), out[1])?;
        patch.dont_apply_twice = Some(format!("fuse-lc-into-post-normalizer@{}", node.id));
        Ok(Some(patch))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("labels"),
        TypeName::Scalar.tensor().named("coefficients"),
        TypeName::Scalar.tensor().named("intercepts"),
        TypeName::Integer.named("multi_class"),
        TypeName::String.named("post_transform"),
    ]
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LinearClassifier,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let labels = ast.konst_variable(format!("{}_labels", node.name), &op.data.labels)?;
    let coefficients =
        ast.konst_variable(format!("{}_coefficients", node.name), &op.data.coefficients)?;

    let post_transform_str = match op.post_transform {
        PostTransformLC::None => "NONE",
        PostTransformLC::Softmax => "SOFTMAX",
        PostTransformLC::Logistic => "LOGISTIC",
    };

    let mut args = vec![input, labels, coefficients];
    let named_args = vec![
        ("multi_class", numeric(op.multi_class as i64)),
        ("post_transform", string(post_transform_str)),
    ];

    if let Some(intercepts) = &op.data.intercepts {
        let intercepts_var = ast.konst_variable(format!("{}_intercepts", node.name), intercepts)?;
        args.push(intercepts_var);
    }

    Ok(Some(invocation("tract_onnx_ml_linear_classifier", &args, &named_args)))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let labels: Arc<Tensor> = invocation.named_arg_as(builder, "labels")?;
    let coefficients: Arc<Tensor> = invocation.named_arg_as(builder, "coefficients")?;
    let intercepts: Option<Arc<Tensor>> = invocation.named_arg_as(builder, "intercepts").ok();
    let multi_class = invocation.named_arg_as(builder, "multi_class")?;
    let post_transform: String = invocation.named_arg_as(builder, "post_transform")?;
    let post_transform = parse_post_transform(&post_transform)?;

    // Precompute eff_e, feat_c, raw and transposed coefficient storage
    let e = labels.len();
    let coef_slice = coefficients.as_slice::<f32>()?;
    let coef_rank = coefficients.rank();
    let (eff_e, feat_c) = if coef_rank == 2 {
        let shape = coefficients.shape();
        (shape[0], shape[1])
    } else if coef_rank == 1 {
        if e == 2 {
            (1, coef_slice.len())
        } else if coef_slice.len() % e == 0 {
            (e, coef_slice.len() / e)
        } else {
            bail!(
                "Cannot infer (classes, features) from coefficients length {} and labels {}",
                coef_slice.len(),
                e
            )
        }
    } else {
        bail!("Unsupported coefficients rank {}", coef_rank)
    };
    if eff_e * feat_c != coef_slice.len() {
        bail!(
            "Inconsistent coefficients size: eff_e={} feat_c={} len={}",
            eff_e,
            feat_c,
            coef_slice.len()
        );
    }
    let mut transposed = Vec::<f32>::with_capacity(coef_slice.len());
    transposed.resize(coef_slice.len(), 0.0);
    for cls in 0..eff_e {
        for f in 0..feat_c {
            transposed[f * eff_e + cls] = coef_slice[cls * feat_c + f];
        }
    }
    let coefficients_raw: Arc<[f32]> = coef_slice.to_vec().into();
    let coefficients_t: Arc<[f32]> = transposed.into();

    // Precompute intercept flavours
    let binary_compact = eff_e == 1 && e == 2;
    let mut bias_scalar: f32 = 0.0;
    let mut intercepts_eff_e: Option<Arc<[f32]>> = None;
    let mut binary_intercepts_2: Option<[f32; 2]> = None;
    if let Some(b) = &intercepts {
        let bsl = b.as_slice::<f32>()?;
        if bsl.len() == 1 {
            bias_scalar = bsl[0];
        } else if bsl.len() == eff_e {
            intercepts_eff_e = Some(bsl.to_vec().into());
        } else if binary_compact && bsl.len() == 2 {
            binary_intercepts_2 = Some([bsl[0], bsl[1]]);
        }
    }

    let data = LinearClassifierData {
        labels: labels.clone(),
        coefficients: coefficients.clone(),
        intercepts: intercepts.clone(),
        eff_e,
        feat_c,
        coefficients_raw,
        coefficients_t,
        binary_compact,
        bias_scalar,
        intercepts_eff_e,
        binary_intercepts_2,
    };
    let op = LinearClassifier { data, multi_class, post_transform };
    builder.wire(op, &[input])
}

#[derive(Debug, Clone, Hash)]
pub struct PostNormalizedLinearClassifier {
    pub lc: LinearClassifier,
    pub norm_kind: crate::ml::normalizer::NormKind,
}

impl Op for PostNormalizedLinearClassifier {
    fn name(&self) -> StaticName {
        "PostNormalizedLinearClassifier".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "norm={:?}, post={:?}, classes={}",
            self.norm_kind,
            self.lc.post_transform,
            self.lc.n_classes()
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for PostNormalizedLinearClassifier {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let ishape = input.shape().to_vec();
        let (outer, _in_c) = match ishape.as_slice() {
            [c] => (1usize, *c),
            [n, c] => (*n, *c),
            other => bail!("Expected input rank 1 or 2, got {} dims", other.len()),
        };
        let eff_e = self.lc.data.eff_e;
        let e = self.lc.n_classes();
        let m_out = if eff_e == 1 && e == 2 { 2 } else { eff_e };
        let mut scores_tensor = unsafe { Tensor::uninitialized::<f32>(&[outer, m_out]) }?;
        let scores_buf = scores_tensor.as_slice_mut::<f32>()?;
        let (_n, _m, argmax) =
            self.lc.eval_scores_normalized_into(&input, scores_buf, true, self.norm_kind)?;
        let labels = if let Some(argmax) = argmax {
            self.lc.labels_from_argmax(&argmax)?
        } else {
            self.lc.compute_labels(&scores_tensor)?
        };
        Ok(tvec!(labels.into_tvalue(), scores_tensor.into_tvalue()))
    }
}

impl TypedOp for PostNormalizedLinearClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.lc.output_facts(inputs)
    }
    as_op!();
}
