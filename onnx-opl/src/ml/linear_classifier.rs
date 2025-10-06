use simsimd::SpatialSimilarity;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use tract_ndarray::prelude::*;
use tract_nnef::internal::*;

// Small reusable per-thread buffer for tiny softmax temps (avoids allocs for e<=8)
thread_local! {
    static SOFTMAX_TMP: RefCell<[f32; 8]> = RefCell::new([0.0; 8]);
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
    pub coefficients: Arc<Tensor>, // original tensor (kept for dump/serialization)
    pub intercepts: Option<Arc<Tensor>>,
    // Pre-computed metadata / optimized storage
    pub eff_e: usize,  // effective number of weight vectors (1 for compact binary)
    pub feat_c: usize, // feature count
    pub coefficients_raw: Arc<[f32]>, // row-major (eff_e, feat_c)
    pub coefficients_t: Arc<[f32]>, // transposed feature-major (feat_c, eff_e)
}

impl Hash for LinearClassifierData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash only canonical sources; derived buffers are excluded.
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
    pub fn n_classes(&self) -> usize {
        self.data.labels.len()
    }
    pub fn labels(&self) -> &Arc<Tensor> {
        &self.data.labels
    }

    #[inline(always)]
    fn matmul_simsimd(input: &ArrayView2<f32>, coef: &ArrayView2<f32>) -> Array2<f32> {
        let n = input.shape()[0];
        let e = coef.shape()[0];
        let mut output = Array2::<f32>::zeros((n, e));
        for i in 0..n {
            if let Some(input_row) = input.row(i).as_slice() {
                for j in 0..e {
                    if let Some(coef_row) = coef.row(j).as_slice() {
                        let dot = f32::dot(input_row, coef_row).unwrap_or(0.0) as f32;
                        output[[i, j]] = dot;
                    }
                }
            }
        }
        output
    }

    #[inline(always)]
    fn softmax_inplace(scores: &mut ArrayViewMut2<f32>) {
        let n = scores.shape()[0];
        let e = scores.shape()[1];
        if e == 1 {
            return;
        }
        if e <= 4 {
            if let Some(data) = scores.as_slice_mut() {
                for row in data.chunks_mut(e) {
                    let mut m = row[0];
                    for &v in &row[1..e] {
                        if v > m {
                            m = v;
                        }
                    }
                    let mut sum = 0.0f32;
                    for v in row.iter_mut().take(e) {
                        *v = (*v - m).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    for v in row.iter_mut().take(e) {
                        *v *= inv;
                    }
                }
                return;
            }
        }
        if let Some(data) = scores.as_slice_mut() {
            for row in data.chunks_mut(e) {
                let mut max_val = row[0];
                for val in &row[1..] {
                    max_val = max_val.max(*val);
                }
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    *val = (*val - max_val).exp();
                    sum += *val;
                }
                let inv_sum = 1.0 / sum;
                for val in row.iter_mut() {
                    *val *= inv_sum;
                }
            }
        } else {
            for i in 0..n {
                unsafe {
                    let mut max_val = *scores.uget([i, 0]);
                    for j in 1..e {
                        let v = *scores.uget([i, j]);
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    let mut sum = 0.0f32;
                    for j in 0..e {
                        let slot = scores.uget_mut([i, j]);
                        *slot = (*slot - max_val).exp();
                        sum += *slot;
                    }
                    let inv = 1.0 / sum;
                    for j in 0..e {
                        let slot = scores.uget_mut([i, j]);
                        *slot *= inv;
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn logistic_inplace(scores: &mut ArrayViewMut2<f32>) {
        for val in scores.iter_mut() {
            *val = 1.0 / (1.0 + (-*val).exp());
        }
    }

    #[inline(always)]
    fn labels_from_argmax(&self, argmax: &[usize]) -> TractResult<Tensor> {
        if self.data.labels.datum_type() == DatumType::String {
            let label_slice = self.data.labels.as_slice::<String>()?;
            let mapped: Vec<String> =
                argmax.iter().map(|&i| label_slice.get(i).cloned().unwrap_or_default()).collect();
            Ok(Tensor::from(tract_ndarray::arr1(&mapped).into_dyn()))
        } else if self.data.labels.datum_type() == DatumType::I64 {
            let label_slice = self.data.labels.as_slice::<i64>()?;
            let mapped: Vec<i64> =
                argmax.iter().map(|&i| label_slice.get(i).copied().unwrap_or(0)).collect();
            Ok(Tensor::from(tract_ndarray::arr1(&mapped).into_dyn()))
        } else {
            bail!("Unsupported label type: {:?}", self.data.labels.datum_type())
        }
    }

    // Fused evaluation path producing scores and (optionally) argmax in one pass.
    // Returns (scores_tensor, argmax_indices)
    fn eval_scores_and_argmax(
        &self,
        input: ArrayViewD<f32>,
        want_argmax: bool,
    ) -> TractResult<(Tensor, Option<Vec<usize>>)> {
        let input_shape = input.shape().to_vec();
        let (_initial_n, initial_c) = if input.ndim() == 1 {
            (1, input.len())
        } else if input.ndim() == 2 {
            (input_shape[0], input_shape[1])
        } else {
            bail!("Expected input rank 1 or 2, got {}", input.ndim())
        };
        let input_2d = if input.ndim() == 1 {
            input.into_shape_with_order((1, initial_c))?
        } else {
            input.into_dimensionality()?
        };
        let n = input_2d.shape()[0];
        let c = input_2d.shape()[1];
        let e = self.n_classes();
        let eff_e = self.data.eff_e;
        let feat_c = self.data.feat_c;
        if feat_c != c {
            bail!("Feature count mismatch: expected {}, got {}", feat_c, c);
        }
        let coef_slice: &[f32] = &self.data.coefficients_raw;
        let intercept_slice_opt: Option<&[f32]> = if let Some(intercepts) = &self.data.intercepts {
            Some(intercepts.as_slice::<f32>()?)
        } else {
            None
        };

        // Fast binary single-vector path when compact (eff_e==1,e==2) and no per-class intercepts
        if eff_e == 1 && e == 2 {
            let per_class_intercepts = intercept_slice_opt.map(|s| s.len() == 2).unwrap_or(false);
            if !per_class_intercepts {
                // For eff_e==1 row-major slice is contiguous weight vector length feat_c
                let w = &coef_slice[..feat_c];
                let bias = intercept_slice_opt
                    .and_then(|s| if s.len() == 1 { Some(s[0]) } else { None })
                    .unwrap_or(0.0);
                let mut scores = Array2::<f32>::zeros((n, 2));
                let mut argmax: Option<Vec<usize>> = want_argmax.then(|| Vec::with_capacity(n));
                match self.post_transform {
                    PostTransformLC::None => {
                        for i in 0..n {
                            let mut z = 0f32;
                            if let Some(row) = input_2d.row(i).as_slice() {
                                for (x, wv) in row.iter().zip(w.iter()) {
                                    z += x * wv;
                                }
                            } else {
                                for k in 0..c {
                                    z += input_2d[[i, k]] * w[k];
                                }
                            }
                            z += bias;
                            scores[[i, 0]] = -z;
                            scores[[i, 1]] = z;
                            if let Some(ref mut a) = argmax {
                                a.push(if z >= 0.0 { 1 } else { 0 });
                            }
                        }
                    }
                    PostTransformLC::Logistic => {
                        for i in 0..n {
                            let mut z = 0f32;
                            if let Some(row) = input_2d.row(i).as_slice() {
                                for (x, wv) in row.iter().zip(w.iter()) {
                                    z += x * wv;
                                }
                            } else {
                                for k in 0..c {
                                    z += input_2d[[i, k]] * w[k];
                                }
                            }
                            z += bias;
                            let p = 1.0 / (1.0 + (-z).exp());
                            scores[[i, 0]] = 1.0 - p;
                            scores[[i, 1]] = p;
                            if let Some(ref mut a) = argmax {
                                a.push(if p >= 0.5 { 1 } else { 0 });
                            }
                        }
                    }
                    PostTransformLC::Softmax => {
                        for i in 0..n {
                            let mut z = 0f32;
                            if let Some(row) = input_2d.row(i).as_slice() {
                                for (x, wv) in row.iter().zip(w.iter()) {
                                    z += x * wv;
                                }
                            } else {
                                for k in 0..c {
                                    z += input_2d[[i, k]] * w[k];
                                }
                            }
                            z += bias;
                            let t = 2.0 * z;
                            let p = 1.0 / (1.0 + (-t).exp());
                            scores[[i, 0]] = 1.0 - p;
                            scores[[i, 1]] = p;
                            if let Some(ref mut a) = argmax {
                                a.push(if p >= 0.5 { 1 } else { 0 });
                            }
                        }
                    }
                }
                return Ok((Tensor::from(scores.into_dyn()), argmax));
            }
        }

        // General path: build ArrayView2 on precomputed row-major slice, use existing matmul then add intercepts + transforms.
        let coef_array = ArrayView2::from_shape((eff_e, feat_c), coef_slice)?;
        let mut scores = Self::matmul_simsimd(&input_2d, &coef_array);
        if let Some(intercepts) = intercept_slice_opt {
            if intercepts.len() == eff_e {
                if let Some(data) = scores.as_slice_mut() {
                    for row in data.chunks_mut(eff_e) {
                        for (v, b) in row.iter_mut().zip(intercepts.iter()) {
                            *v += *b;
                        }
                    }
                } else {
                    for i in 0..n {
                        for j in 0..eff_e {
                            scores[[i, j]] += intercepts[j];
                        }
                    }
                }
            } else if eff_e == 1 && intercepts.len() == 1 {
                if let Some(data) = scores.as_slice_mut() {
                    for v in data.iter_mut() {
                        *v += intercepts[0];
                    }
                } else {
                    for i in 0..n {
                        scores[[i, 0]] += intercepts[0];
                    }
                }
            }
        }
        match self.post_transform {
            PostTransformLC::None => {}
            PostTransformLC::Softmax => {
                Self::softmax_inplace(&mut scores.view_mut());
            }
            PostTransformLC::Logistic => {
                Self::logistic_inplace(&mut scores.view_mut());
            }
        }
        if eff_e == 1 && e == 2 {
            // expand
            let compact = scores;
            let mut full = Array2::<f32>::zeros((n, 2));
            if let (Some(src), Some(dst)) = (compact.as_slice(), full.as_slice_mut()) {
                match self.post_transform {
                    PostTransformLC::None => {
                        for (v, row) in src.iter().zip(dst.chunks_mut(2)) {
                            row[0] = -*v;
                            row[1] = *v;
                        }
                    }
                    _ => {
                        for (v, row) in src.iter().zip(dst.chunks_mut(2)) {
                            row[0] = 1.0 - *v;
                            row[1] = *v;
                        }
                    }
                }
            } else {
                match self.post_transform {
                    PostTransformLC::None => {
                        for i in 0..n {
                            full[[i, 0]] = -compact[[i, 0]];
                            full[[i, 1]] = compact[[i, 0]];
                        }
                    }
                    _ => {
                        for i in 0..n {
                            full[[i, 0]] = 1.0 - compact[[i, 0]];
                            full[[i, 1]] = compact[[i, 0]];
                        }
                    }
                }
            }
            if let Some(intercepts) = intercept_slice_opt {
                if intercepts.len() == 2 {
                    if let Some(data) = full.as_slice_mut() {
                        for row in data.chunks_mut(2) {
                            row[0] += intercepts[0];
                            row[1] += intercepts[1];
                        }
                    } else {
                        for i in 0..n {
                            full[[i, 0]] += intercepts[0];
                            full[[i, 1]] += intercepts[1];
                        }
                    }
                }
            }
            let mut argmax = want_argmax.then(|| Vec::with_capacity(n));
            if let Some(ref mut a) = argmax {
                if let Some(data) = full.as_slice() {
                    for row in data.chunks(2) {
                        a.push(if row[1] >= row[0] { 1 } else { 0 });
                    }
                } else {
                    for i in 0..n {
                        a.push(if full[[i, 1]] >= full[[i, 0]] { 1 } else { 0 });
                    }
                }
            }
            return Ok((Tensor::from(full.into_dyn()), argmax));
        }
        let mut argmax = want_argmax.then(|| Vec::with_capacity(n));
        if let Some(ref mut a) = argmax {
            for i in 0..n {
                let mut best = 0usize;
                let mut best_v = scores[[i, 0]];
                for j in 1..e {
                    let v = scores[[i, j]];
                    if v > best_v {
                        best_v = v;
                        best = j;
                    }
                }
                a.push(best);
            }
        }
        Ok((Tensor::from(scores.into_dyn()), argmax))
    }

    pub fn eval_scores(&self, input: ArrayViewD<f32>) -> TractResult<Tensor> {
        let (scores, _) = self.eval_scores_and_argmax(input, false)?;
        Ok(scores)
    }
    pub fn eval(&self, input: ArrayViewD<f32>) -> TractResult<(Tensor, Tensor)> {
        let (scores, argmax) = self.eval_scores_and_argmax(input, true)?;
        let labels = if let Some(argmax) = argmax {
            self.labels_from_argmax(&argmax)?
        } else {
            self.compute_labels(&scores)?
        };
        Ok((labels, scores))
    }

    fn compute_labels(&self, scores: &Tensor) -> TractResult<Tensor> {
        let scores_view = scores.to_array_view::<f32>()?;
        let scores_2d: ArrayView2<f32> = scores_view.into_dimensionality()?;
        let n = scores_2d.shape()[0];

        let mut argmax_indices = Vec::with_capacity(n);
        for i in 0..n {
            let mut max_idx = 0;
            let mut max_val = scores_2d[[i, 0]];
            for j in 1..scores_2d.shape()[1] {
                if scores_2d[[i, j]] > max_val {
                    max_val = scores_2d[[i, j]];
                    max_idx = j;
                }
            }
            argmax_indices.push(max_idx);
        }

        let labels = if self.data.labels.datum_type() == DatumType::String {
            let label_slice = self.data.labels.as_slice::<String>()?;
            let mapped: Vec<String> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).cloned().unwrap_or_default())
                .collect();
            Tensor::from(tract_ndarray::arr1(&mapped).into_dyn())
        } else if self.data.labels.datum_type() == DatumType::I64 {
            let label_slice = self.data.labels.as_slice::<i64>()?;
            let mapped: Vec<i64> = argmax_indices
                .iter()
                .map(|&idx| label_slice.get(idx).copied().unwrap_or(0))
                .collect();
            Tensor::from(tract_ndarray::arr1(&mapped).into_dyn())
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
        let input = input.to_array_view::<f32>()?;
        let (labels, scores) = self.eval(input)?;
        Ok(tvec!(labels.into_tvalue(), scores.into_tvalue()))
    }
}

impl TypedOp for LinearClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        let label_dt = self.data.labels.datum_type();
        Ok(tvec!(label_dt.fact(&[n.clone()]), f32::fact(&[n.clone(), self.n_classes().into()])))
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
    let intercepts = invocation.named_arg_as(builder, "intercepts").ok();
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

    let data = LinearClassifierData {
        labels: labels.clone(),
        coefficients: coefficients.clone(),
        intercepts: intercepts.clone(),
        eff_e,
        feat_c,
        coefficients_raw,
        coefficients_t,
    };
    let op = LinearClassifier { data, multi_class, post_transform };
    builder.wire(op, &[input])
}
