use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;

// Import types from OPL to avoid duplication
pub use tract_onnx_opl::ml::linear_classifier::{PostTransformLC, parse_post_transform};

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LinearClassifier", linear_classifier);
}

fn parse_labels(node: &NodeProto) -> TractResult<Arc<Tensor>> {
    if let Some(strings) = node.get_attr_opt_tvec::<&str>("classlabels_strings")? {
        Ok(rctensor1(&strings.iter().map(|s| s.to_string()).collect::<Vec<String>>()))
    } else if let Some(ints) = node.get_attr_opt_slice::<i64>("classlabels_ints")? {
        Ok(rctensor1(ints))
    } else {
        bail!("one of classlabels_strings or classlabels_ints must be set")
    }
}

fn linear_classifier(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let labels = parse_labels(node)?;

    let coefs = node.get_attr_tvec::<f32>("coefficients")?;
    let intercepts = node.get_attr_opt_tvec::<f32>("intercepts")?.unwrap_or(tvec!());
    let multi_class: i64 = node.get_attr_opt("multi_class")?.unwrap_or(0);
    let post_t = parse_post_transform(node.get_attr_opt("post_transform")?.unwrap_or("NONE"))?;

    Ok((
        expand(LinearClassifier {
            labels,
            coefficients: rctensor1(&coefs),
            intercepts: if intercepts.is_empty() { None } else { Some(rctensor1(&intercepts)) },
            multi_class: multi_class as i32,
            post_transform: post_t,
        }),
        vec![],
    ))
}

#[derive(Debug, Clone, Hash)]
pub struct LinearClassifier {
    pub labels: Arc<Tensor>,
    pub coefficients: Arc<Tensor>, // shape [E*C] or [C] for binary OvR
    pub intercepts: Option<Arc<Tensor>>, // shape [E] or [1]
    pub multi_class: i32,          // 0 OvR, 1 multinomial
    pub post_transform: PostTransformLC,
}

impl Expansion for LinearClassifier {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "LinearClassifier".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "multi_class={}, post_transform={:?}",
            self.multi_class, self.post_transform
        )])
    }
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 2)?;
        s.equals(&outputs[0].datum_type, self.labels.datum_type())?;
        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].datum_type, DatumType::F32)?;
        s.equals(&outputs[1].rank, 2)?;
        s.equals(&outputs[1].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].shape[1], self.labels.len().to_dim())?;
        Ok(())
    }
    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let e = self.labels.len();
        let coef_slice = self.coefficients.as_slice::<f32>()?;
        let total = coef_slice.len();
        let (eff_e, feat_c) = if total % e == 0 {
            (e, total / e)
        } else if e == 2 {
            (1, total)
        } else {
            bail!(
                "Cannot infer (classes, features) from coefficients length {} and labels {}",
                total,
                e
            )
        };
        if eff_e * feat_c != total {
            bail!("Inconsistent coefficients size eff_e={} feat_c={} len={}", eff_e, feat_c, total);
        }
        let mut transposed = vec![0f32; total];
        for cls in 0..eff_e {
            for f in 0..feat_c {
                transposed[f * eff_e + cls] = coef_slice[cls * feat_c + f];
            }
        }
        let coefficients_raw: Arc<[f32]> = coef_slice.to_vec().into();
        let coefficients_t: Arc<[f32]> = transposed.into();
        let data = tract_onnx_opl::ml::linear_classifier::LinearClassifierData {
            labels: self.labels.clone(),
            coefficients: self.coefficients.clone(),
            intercepts: self.intercepts.clone(),
            eff_e,
            feat_c,
            coefficients_raw,
            coefficients_t,
        };
        let outputs = model.wire_node(
            format!("{prefix}"),
            tract_onnx_opl::ml::linear_classifier::LinearClassifier {
                data,
                multi_class: self.multi_class as i64,
                post_transform: self.post_transform,
            },
            inputs,
        )?;
        Ok(tvec!(outputs[0], outputs[1]))
    }
    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid_scalar(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn multiclass_none_int_labels() {
        // 3 classes, 2 features
        // class0: [1, 0], class1: [0, 1], class2: [-1, -1]
        let labels = rctensor1(&[10i64, 20, 30]);
        let coefs = rctensor1(&[1f32, 0., 0., 1., -1., -1.]);
        let op = LinearClassifier {
            labels: labels.clone(),
            coefficients: coefs,
            intercepts: None,
            multi_class: 1,
            post_transform: PostTransformLC::None,
        };
        let x = tensor2(&[[1f32, 0.], [0., 1.]]);
        let outputs = expand(op).eval(tvec!(x.into_tvalue())).unwrap();
        let y = outputs[0].clone().into_tensor();
        let z = outputs[1].clone().into_tensor();

        // Y should map to labels 10 then 20
        assert_eq!(y.as_slice::<i64>().unwrap(), &[10i64, 20i64]);

        // Z should be raw scores [[1,0,-1],[0,1,-1]]
        let expected = tensor2(&[[1f32, 0., -1.], [0., 1., -1.]]);
        expected.close_enough(&z, true).unwrap();
    }

    #[test]
    fn binary_none_singlevec_string_labels() {
        // 2 classes with single-vector coefficients, NONE post-transform
        // s = 2*x0 - 1*x1; Z = [-s, s]
        let labels = rctensor1(&["no".to_string(), "yes".to_string()]);
        let coefs = rctensor1(&[2f32, -1.0]);
        let op = LinearClassifier {
            labels: labels.clone(),
            coefficients: coefs,
            intercepts: None,
            multi_class: 0,
            post_transform: PostTransformLC::None,
        };
        let x = tensor2(&[[1f32, 0.], [0., 3.]]);
        let outputs = expand(op).eval(tvec!(x.into_tvalue())).unwrap();
        let y = outputs[0].clone().into_tensor();
        let z = outputs[1].clone().into_tensor();

        // First row s=2 -> class 1 ("yes"), second row s=-3 -> class 0 ("no")
        let y_vals = y.as_slice::<String>().unwrap();
        assert_eq!(y_vals, &["yes".to_string(), "no".to_string()]);

        // Z should be [[-2, 2], [3, -3]]
        let expected = tensor2(&[[-2f32, 2.], [3., -3.]]);
        expected.close_enough(&z, true).unwrap();
    }

    #[test]
    fn binary_logistic_rank1_input() {
        // 2 classes, single feature, logistic post-transform, rank-1 input
        // s = 1 * x; p = sigmoid(s); Z = [1-p, p]
        let labels = rctensor1(&[0i64, 1i64]);
        let coefs = rctensor1(&[1f32]);
        let op = LinearClassifier {
            labels: labels.clone(),
            coefficients: coefs,
            intercepts: Some(rctensor1(&[0f32])),
            multi_class: 0,
            post_transform: PostTransformLC::Logistic,
        };
        let x = tensor1(&[2f32]); // rank-1 input should be accepted
        let outputs = expand(op).eval(tvec!(x.into_tvalue())).unwrap();
        let y = outputs[0].clone().into_tensor();
        let z = outputs[1].clone().into_tensor();

        // Argmax should be class 1
        assert_eq!(y.as_slice::<i64>().unwrap(), &[1i64]);

        let p = sigmoid_scalar(2.0);
        let expected = tensor2(&[[1.0 - p, p]]);
        expected.close_enough(&z, true).unwrap();
    }
}
