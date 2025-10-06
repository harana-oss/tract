use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Normalizer", normalizer);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NormKind {
    Max,
    L1,
    L2,
}

fn parse_norm_kind(s: &str) -> TractResult<NormKind> {
    match s.to_ascii_uppercase().as_str() {
        "MAX" => Ok(NormKind::Max),
        "L1" => Ok(NormKind::L1),
        "L2" => Ok(NormKind::L2),
        other => bail!("Invalid norm kind: {}", other),
    }
}

#[derive(Debug, Clone, Hash)]
struct Normalizer {
    kind: NormKind,
}

fn normalizer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let norm: String = node.get_attr_opt("norm")?.unwrap_or_else(|| "MAX".to_string());
    let kind = parse_norm_kind(&norm)?;
    Ok((expand(Normalizer { kind }), vec![]))
}

impl Expansion for Normalizer {
    fn name(&self) -> StaticName {
        "Normalizer".into()
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
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        // Use the OPL version for the core normalizer computation
        let kind_opl = match self.kind {
            NormKind::Max => tract_onnx_opl::ml::normalizer::NormKind::Max,
            NormKind::L1 => tract_onnx_opl::ml::normalizer::NormKind::L1,
            NormKind::L2 => tract_onnx_opl::ml::normalizer::NormKind::L2,
        };

        let y = model.wire_node(
            format!("{prefix}.normalizer"),
            tract_onnx_opl::ml::normalizer::Normalizer { kind: kind_opl },
            inputs,
        )?;

        Ok(tvec!(y[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }
}
