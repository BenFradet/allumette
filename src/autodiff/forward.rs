use std::rc::Rc;

use crate::{
    autodiff::{context::Context, history::History},
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    ops::{binary::Binary, function::Function, unary::Unary},
    tensor::Tensor,
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

pub struct Forward;

impl Forward {
    pub fn binary<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
        b: impl Binary<E, BT, T> + 'static,
        lhs: Tensor<E, BT, T>,
        rhs: Tensor<E, BT, T>,
    ) -> Tensor<E, BT, T> {
        let res = b.forward(&lhs.data, &rhs.data);
        let ctx = Context::default()
            .fst(lhs.data.clone())
            .snd(rhs.data.clone());
        let new_history = if lhs.is_constant && rhs.is_constant {
            History::default()
        } else {
            History::default()
                .last_fn(Function::B(Rc::new(b)))
                .context(ctx)
                .push_input(lhs)
                .push_input(rhs)
        };
        Tensor::new(res, new_history)
    }

    pub fn unary<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
        u: impl Unary<E, BT, T> + 'static,
        a: Tensor<E, BT, T>,
    ) -> Tensor<E, BT, T> {
        let res = u.forward(&a.data);
        let ctx = Context::default().fst(a.data.clone());
        let new_history = if a.is_constant {
            History::default()
        } else {
            History::default()
                .last_fn(Function::U(Rc::new(u)))
                .context(ctx)
                .push_input(a)
        };
        Tensor::new(res, new_history)
    }
}
