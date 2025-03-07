use std::rc::Rc;

use crate::{
    autodiff::context::Context,
    function::{binary::Binary, function::Function, unary::Unary},
};

use super::{tensor::Tensor, tensor_data::TensorData, tensor_history::TensorHistory};

pub struct Forward;

impl Forward {
    pub fn binary(b: impl Binary<TensorData> + 'static, lhs: Tensor, rhs: Tensor) -> Tensor {
        let res = b.forward(lhs.data.clone(), rhs.data.clone());
        let ctx = Context::default()
            .fst(lhs.data.clone())
            .snd(rhs.data.clone());
        let new_history = TensorHistory::default()
            .last_fn(Function::B(Rc::new(b)))
            .context(ctx)
            .push_input(lhs)
            .push_input(rhs);
        Tensor::new(res, new_history)
    }

    pub fn unary(u: impl Unary<TensorData> + 'static, a: Tensor) -> Tensor {
        let res = u.forward(a.data.clone());
        let ctx = Context::default().fst(a.data.clone());
        let new_history = TensorHistory::default()
            .last_fn(Function::U(Rc::new(u)))
            .context(ctx)
            .push_input(a);
        Tensor::new(res, new_history)
    }
}
