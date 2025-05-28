use std::rc::Rc;

use crate::{
    autodiff::{context::Context, history::History},
    backend::{backend::TensorBackend, backend_type::TensorBackendType},
    ops::{binary::Binary, function::Function, unary::Unary},
    tensor::Tensor,
};

pub struct Forward;

impl Forward {
    pub fn binary<BT: TensorBackendType, T: TensorBackend<BT>>(
        b: impl Binary<BT, T> + 'static,
        lhs: Tensor<BT, T>,
        rhs: Tensor<BT, T>,
    ) -> Tensor<BT, T> {
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

    pub fn unary<BT: TensorBackendType, T: TensorBackend<BT>>(
        u: impl Unary<BT, T> + 'static,
        a: Tensor<BT, T>,
    ) -> Tensor<BT, T> {
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
