use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::*;
use tract_hir::internal::*;

// Re-export OPL types
pub use tract_onnx_opl::ml::linear_regressor::{PostTransformLR, parse_post_transform_lr};

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LinearRegressor", linear_regressor);
}

fn linear_regressor(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    // targets attribute (default 1)
    let targets: i64 = node.get_attr_opt("targets")?.unwrap_or(1);
    node.expect(targets > 0, "targets must be > 0")?;
    let targets: usize =
        usize::try_from(targets).map_err(|_| format_err!("targets out of range"))?;

    // coefficients: flat vector length = targets * features (or features if targets==1)
    let coefs = node.get_attr_tvec::<f32>("coefficients")?;
    node.expect(!coefs.is_empty(), "coefficients not empty")?;
    node.expect(coefs.len() % targets == 0, "coefficients length multiple of targets")?;

    // optional intercepts: length == targets or 1
    let intercepts = node.get_attr_opt_tvec::<f32>("intercepts")?.unwrap_or(tvec!());

    // post_transform (NONE by default)
    let post_t = parse_post_transform_lr(node.get_attr_opt("post_transform")?.unwrap_or("NONE"))?;

    Ok((
        expand(LinearRegressorBridge {
            coefficients: rctensor1(&coefs),
            intercepts: if intercepts.is_empty() { None } else { Some(rctensor1(&intercepts)) },
            targets,
            post_transform: post_t,
        }),
        vec![],
    ))
}

#[derive(Debug, Clone, Hash)]
pub struct LinearRegressorBridge {
    pub coefficients: Arc<Tensor>,       // flat tensor (targets * features)
    pub intercepts: Option<Arc<Tensor>>, // shape [targets] or [1]
    pub targets: usize,
    pub post_transform: PostTransformLR,
}

impl Expansion for LinearRegressorBridge {
    fn name(&self) -> StaticName {
        "LinearRegressor".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("targets={}, post_transform={:?}", self.targets, self.post_transform)])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, DatumType::F32)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?; // batch dimension
        s.equals(&outputs[0].shape[1], self.targets.to_dim())?; // targets
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        // Build OPL data structure (computes metadata, transposed buffers, etc.)
        let data = tract_onnx_opl::ml::linear_regressor::LinearRegressorData::new(
            self.coefficients.clone(),
            self.intercepts.clone(),
            self.targets,
        )?;
        let outputs = model.wire_node(
            format!("{prefix}"),
            tract_onnx_opl::ml::linear_regressor::LinearRegressorOp {
                data,
                post_transform: self.post_transform,
            },
            inputs,
        )?;
        Ok(tvec!(outputs[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }
}
