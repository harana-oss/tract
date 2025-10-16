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
        // Precompute intercept flavours to match OPL runtime expectations
        let binary_compact = eff_e == 1 && e == 2;
        let mut bias_scalar: f32 = 0.0;
        let mut intercepts_eff_e: Option<Arc<[f32]>> = None;
        let mut binary_intercepts_2: Option<[f32; 2]> = None;
        if let Some(b) = &self.intercepts {
            let bsl = b.as_slice::<f32>()?;
            if bsl.len() == 1 {
                bias_scalar = bsl[0];
            } else if bsl.len() == eff_e {
                intercepts_eff_e = Some(bsl.to_vec().into());
            } else if binary_compact && bsl.len() == 2 {
                binary_intercepts_2 = Some([bsl[0], bsl[1]]);
            }
        }
        let data = tract_onnx_opl::ml::linear_classifier::LinearClassifierData {
            labels: self.labels.clone(),
            coefficients: self.coefficients.clone(),
            intercepts: self.intercepts.clone(),
            eff_e,
            feat_c,
            coefficients_raw,
            coefficients_t,
            labels_i64: match self.labels.datum_type() {
                DatumType::I64 => Some(self.labels.as_slice::<i64>()?.to_vec().into()),
                _ => None,
            },
            labels_str: match self.labels.datum_type() {
                DatumType::String => Some(self.labels.as_slice::<String>()?.to_vec().into()),
                _ => None,
            },
            labels_are_iota: match self.labels.datum_type() {
                DatumType::I64 => {
                    let sl = self.labels.as_slice::<i64>()?;
                    sl.iter().enumerate().all(|(i, &v)| v == i as i64)
                }
                _ => false,
            },
            packed_tiled: None,
            prefer_transposed: eff_e >= 4 && feat_c >= 16,
            m_out: if eff_e == 1 && e == 2 { 2 } else { eff_e },
            binary_compact,
            bias_scalar,
            intercepts_eff_e,
            binary_intercepts_2,
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
