use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::array::{Slice, TypedConcat};
use tract_hir::tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_onnx_opl::ml::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LinearClassifier", linear_classifier);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransform {
    Softmax,
    Logistic,
}

pub fn parse_post_transform(s: &str) -> TractResult<Option<PostTransform>> {
    match s {
        "NONE" => Ok(None),
        "SOFTMAX" => Ok(Some(PostTransform::Softmax)),
        "LOGISTIC" => Ok(Some(PostTransform::Logistic)),
        "PROBIT" | "SOFTMAX_ZERO" => bail!("PROBIT and SOFTMAX_ZERO unsupported"),
        _ => bail!("Invalid post transform: {}", s),
    }
}

fn parse_class_data(node: &NodeProto) -> TractResult<Arc<Tensor>> {
    let ints = node.get_attr_opt_slice::<i64>("classlabels_ints")?;
    let strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?;
    match (ints, strs) {
        (Some(n), None) => Ok(rctensor1(n)),
        (None, Some(n)) => Ok(rctensor1(&n.iter().map(|d| d.to_string()).collect::<Vec<_>>())),
        (None, None) => bail!("cannot find neither 'classlabels_ints' not 'classlabels_strings'"),
        (Some(_), Some(_)) => {
            bail!("only one of 'classlabels_ints' and 'classlabels_strings' can be set")
        }
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearClassifier {
    pub class_labels: Arc<Tensor>,
    pub coefficients: Arc<Tensor>,
    pub intercepts: Option<Arc<Tensor>>,
    pub post_transform: Option<PostTransform>,
    pub binary_result_layout: bool,
}

fn linear_classifier(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let class_labels = parse_class_data(node)?;
    let n_classes = class_labels.len();
    let multi_class: i64 = node.get_attr_opt("multi_class")?.unwrap_or(0);
    let raw_coeffs: Vec<f32> = node.get_attr_vec("coefficients")?;
    node.expect(!raw_coeffs.is_empty(), "coefficients not empty")?;

    let (e_prime, binary_result_layout) = if raw_coeffs.len() % n_classes == 0 {
        (n_classes, false)
    } else if n_classes == 2 && multi_class == 0 {
        // OvR binary with a single set of coefficients
        (1, true)
    } else {
        bail!(
            "coefficients length {} not compatible with number of classes {}",
            raw_coeffs.len(),
            n_classes
        )
    };

    let c = raw_coeffs.len() / e_prime;
    // Build as [E', C], then transpose to contiguous [C, E'] for better locality and to avoid
    // runtime transpose. This also enables SGEMV fast-path when E'==1.
    let coeffs_ec = tensor1(&raw_coeffs).into_shape(&[e_prime, c])?;
    let coefficients = coeffs_ec.permute_axes(&[1, 0])?.into_arc_tensor();

    let intercepts: Option<Vec<f32>> = node.get_attr_opt_vec("intercepts")?;
    let intercepts = match intercepts {
        Some(v) => {
            node.expect(v.len() == e_prime, "intercepts length matches number of models")?;
            Some(rctensor1(&v))
        }
        None => None,
    };

    let post_transform =
        node.get_attr_opt("post_transform")?.map(parse_post_transform).transpose()?.unwrap_or(None);

    Ok((
        expand(LinearClassifier {
            class_labels,
            coefficients,
            intercepts,
            post_transform,
            binary_result_layout,
        }),
        vec![],
    ))
}

impl Expansion for LinearClassifier {
    fn name(&self) -> StaticName {
        "LinearClassifier".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 2)?;

        s.equals(&outputs[0].datum_type, self.class_labels.datum_type())?;
        s.equals(&outputs[1].datum_type, DatumType::F32)?;

        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[1].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].shape[1], self.class_labels.len().to_dim())?;

        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::nn::*;

        let mut x = inputs[0];
        if model.outlet_fact(x)?.rank() == 1 {
            x = model.wire_node(format!("{prefix}.add_batch_axis"), AxisOp::Add(0), &[x])?[0];
        }

        if model.outlet_fact(x)?.datum_type != f32::datum_type() {
            x = model.wire_node(
                format!("{prefix}.to_f32"),
                tract_core::ops::cast::cast(f32::datum_type()),
                &[x],
            )?[0];
        }

        let w = model.add_const(format!("{prefix}.coefficients"), self.coefficients.clone())?;
        let mut scores = {
            model.wire_node(
                format!("{prefix}.matmul"),
                PrefixMatMul {
                    transpose_a: false,
                    transpose_b: false,
                    transpose_c: false,
                    quantize_output: None,
                },
                [x, w].as_ref(),
            )?
        };

        if let Some(intercepts) = self.intercepts.as_deref() {
            let bias = intercepts.clone().broadcast_into_rank(2)?.into_arc_tensor();
            let bias = model.add_const(format!("{prefix}.intercepts"), bias)?;
            scores = model.wire_node(
                format!("{prefix}.add_bias"),
                tract_core::ops::math::add(),
                &[scores[0], bias],
            )?;
        }

        match self.post_transform {
            None => (),
            Some(PostTransform::Softmax) => {
                scores = model.wire_node(
                    format!("{prefix}.softmax"),
                    tract_core::ops::nn::Softmax {
                        axes: tvec![1],
                        quant_output_dt: None,
                        kind: tract_core::ops::nn::SoftmaxKind::Softmax(
                            tract_core::ops::nn::SoftmaxExp::Libc,
                        ),
                    },
                    &scores,
                )?;
            }
            Some(PostTransform::Logistic) => {
                scores = model.wire_node(
                    format!("{prefix}.logistic"),
                    tract_core::ops::nn::sigmoid(),
                    &scores,
                )?;
            }
        }

        let mut final_scores = scores.clone();
        if self.binary_result_layout {
            let s1 =
                model.wire_node(format!("{prefix}.binary.slice"), Slice::new(1, 0, 1), &scores)?;
            let one = model.add_const(prefix.to_string() + ".one", rctensor2(&[[1f32]]))?;
            let complement = model.wire_node(
                format!("{prefix}.binary.complement"),
                tract_core::ops::math::sub(),
                &[one, s1[0]],
            )?;
            final_scores = model.wire_node(
                format!("{prefix}.binary.concat"),
                TypedConcat::new(1),
                &[complement[0], s1[0]],
            )?;
        }

        let winners = model.wire_node(
            format!("{prefix}.argmax"),
            tract_core::ops::nn::Reduce::new(tvec!(1), tract_core::ops::nn::Reducer::ArgMax(false)),
            &final_scores,
        )?;
        let reduced = model.wire_node(
            format!("{prefix}.rm_axis"),
            tract_core::ops::change_axes::AxisOp::Rm(1),
            &winners,
        )?;
        let casted = model.wire_node(
            format!("{prefix}.casted"),
            tract_core::ops::cast::cast(i32::datum_type()),
            &reduced,
        )?;
        let labels = model.wire_node(
            format!("{prefix}.labels"),
            DirectLookup::new(
                self.class_labels.clone(),
                Tensor::zero_dt(self.class_labels.datum_type(), &[])?.into_arc_tensor(),
            )?,
            &casted,
        )?[0];

        Ok(tvec!(labels, final_scores[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }
}
