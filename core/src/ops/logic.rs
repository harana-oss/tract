#![allow(clippy::bool_comparison)]
#![allow(clippy::unnecessary_cast)]

mod comparison;
mod ite;
pub use comparison::Comp;
pub use ite::IfThenElse;

use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

bin_to_super_type!(and, And,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 && b as i64 != 0) as _);
bin_to_super_type!(or, Or,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 || b as i64 != 0) as _);
bin_to_super_type!(xor, Xor, /*flip: commute, */ [bool] => |c, &a, &b| *c = a ^ b);

element_wise!(not, Not, [bool] => |_, vs| {
    vs.iter_mut().for_each(|a| *a = !*a);
    Ok(())
});

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Iff;

impl Iff {
    pub unsafe fn eval_t<T: Datum>(
        cond: &ArrayViewD<bool>,
        out: &mut Tensor,
        t: &Tensor,
        f: &Tensor,
    ) {
        unsafe {
            // Prepare views once
            let mut out_view = out.to_array_view_mut_unchecked::<T>();
            let t_view = t.to_array_view_unchecked::<T>();
            let f_view = f.to_array_view_unchecked::<T>();
            let out_shape: Vec<usize> = out_view.shape().to_vec();

            // Helper: copy src -> out with broadcasting (or memcpy when possible)
            #[inline(always)]
            unsafe fn copy_into<T: Datum>(out_view: &mut ArrayViewMutD<T>, src: &ArrayViewD<T>) {
                if src.shape() == out_view.shape() {
                    if let (Some(dst), Some(s)) =
                        (out_view.as_slice_memory_order_mut(), src.as_slice_memory_order())
                    {
                        dst.clone_from_slice(s);
                        return;
                    }
                }
                Zip::from(out_view.view_mut())
                    .and_broadcast(src.view())
                    .for_each(|o, s| *o = s.clone());
            }

            // Fast path 1: cond is a single element (broadcasted scalar)
            if cond.len() == 1 {
                let c0 = *cond.iter().next().unwrap_or(&false);
                if c0 {
                    copy_into(&mut out_view, &t_view);
                } else {
                    copy_into(&mut out_view, &f_view);
                }
                return;
            }

            // Fast path 2: cond is all true or all false (any shape)
            let mut any_false = false;
            for &c in cond.iter() {
                if !c {
                    any_false = true;
                    break;
                }
            }
            if !any_false {
                copy_into(&mut out_view, &t_view);
                return;
            }
            let mut any_true = false;
            for &c in cond.iter() {
                if c {
                    any_true = true;
                    break;
                }
            }
            if !any_true {
                copy_into(&mut out_view, &f_view);
                return;
            }

            // Fast path 3: all have identical shape and contiguous layout -> tight loop
            if cond.shape() == out_shape.as_slice()
                && t_view.shape() == out_shape.as_slice()
                && f_view.shape() == out_shape.as_slice()
            {
                if let (Some(cnd), Some(ts), Some(fs), Some(dst)) = (
                    cond.as_slice_memory_order(),
                    t_view.as_slice_memory_order(),
                    f_view.as_slice_memory_order(),
                    out_view.as_slice_memory_order_mut(),
                ) {
                    for i in 0..dst.len() {
                        dst[i] = if cnd[i] { ts[i].clone() } else { fs[i].clone() };
                    }
                    return;
                }
            }

            // Fast path 4: t is scalar
            if t_view.len() == 1 {
                let t0 = t_view.iter().next().unwrap().clone();
                if cond.shape() == out_shape.as_slice() && f_view.shape() == out_shape.as_slice() {
                    if let (Some(cnd), Some(fs), Some(dst)) = (
                        cond.as_slice_memory_order(),
                        f_view.as_slice_memory_order(),
                        out_view.as_slice_memory_order_mut(),
                    ) {
                        for i in 0..dst.len() {
                            dst[i] = if cnd[i] { t0.clone() } else { fs[i].clone() };
                        }
                        return;
                    }
                }
                Zip::from(out_view.view_mut())
                    .and_broadcast(cond.view())
                    .and_broadcast(f_view.view())
                    .for_each(|o, &c, fv| *o = if c { t0.clone() } else { fv.clone() });
                return;
            }

            // Fast path 5: f is scalar
            if f_view.len() == 1 {
                let f0 = f_view.iter().next().unwrap().clone();
                if cond.shape() == out_shape.as_slice() && t_view.shape() == out_shape.as_slice() {
                    if let (Some(cnd), Some(ts), Some(dst)) = (
                        cond.as_slice_memory_order(),
                        t_view.as_slice_memory_order(),
                        out_view.as_slice_memory_order_mut(),
                    ) {
                        for i in 0..dst.len() {
                            dst[i] = if cnd[i] { ts[i].clone() } else { f0.clone() };
                        }
                        return;
                    }
                }
                Zip::from(out_view.view_mut())
                    .and_broadcast(cond.view())
                    .and_broadcast(t_view.view())
                    .for_each(|o, &c, tv| *o = if c { tv.clone() } else { f0.clone() });
                return;
            }

            // Fallback: generic broadcasting with branching per element
            Zip::from(out_view)
                .and_broadcast(cond.view())
                .and_broadcast(t_view)
                .and_broadcast(f_view)
                .for_each(|r, c, t, f| *r = if *c { t.clone() } else { f.clone() })
        }
    }
}

impl Op for Iff {
    fn name(&self) -> StaticName {
        "Iff".into()
    }
    op_as_typed_op!();
}

impl EvalOp for Iff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (cond, t, f) = args_3!(inputs);
        anyhow::ensure!(t.datum_type() == f.datum_type());
        let shape: TVec<usize> = multi_broadcast(&[cond.shape(), t.shape(), f.shape()])?;
        unsafe {
            let mut result = Tensor::uninitialized_dt(t.datum_type(), &shape)?;
            let cond = cond.to_array_view::<bool>()?;
            dispatch_datum_by_size!(Self::eval_t(t.datum_type())(&cond, &mut result, &t, &f));
            Ok(tvec!(result.into_tvalue()))
        }
    }
}

impl TypedOp for Iff {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 3, "Iff expects 3 intputs.");
        if inputs[1].datum_type != inputs[2].datum_type {
            bail!("Then and else tensors type mismatch ({:?} and {:?}).", inputs[1], inputs[2]);
        }
        if inputs[0].rank() != inputs[1].rank() || inputs[0].rank() != inputs[2].rank() {
            bail!("Inconsistent ranks, {:?}", inputs);
        }
        let shape = multi_broadcast(&[
            inputs[0].shape.to_tvec(),
            inputs[1].shape.to_tvec(),
            inputs[2].shape.to_tvec(),
        ])
        .unwrap();
        Ok(tvec!(inputs[1].datum_type.fact(shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }
}

bin_to_super_type!(bitand, BitAnd,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a & b);
bin_to_super_type!(bitor, BitOr,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a | b);
bin_to_super_type!(bitxor, BitXor,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a ^ b);

element_wise!(bitnot, BitNot, [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = !*x);
    Ok(())
});
