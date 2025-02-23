use std::rc::Rc;

use crate::{autodiff::{context::Context, history::History}, function::{binary::Binary, function::Function, unary::Unary}};

use super::{tensor::Tensor, tensor_data::TensorData};

pub struct Forward;

impl Forward {
    pub fn binary(b: impl Binary<TensorData> + 'static, lhs: TensorData, rhs: TensorData) -> Tensor {
        let res = b.forward(lhs.clone(), rhs.clone());
        let ctx = Context::default().fst(lhs.clone()).snd(rhs.clone());
        let new_history = History::default()
            .last_fn(Function::B(Rc::new(b)))
            .context(ctx)
            .push_input(lhs)
            .push_input(rhs);
        Tensor::new(res, new_history)
    }

    pub fn unary(u: impl Unary<TensorData> + 'static, a: TensorData) -> Tensor {
        let res = u.forward(a.clone());
        let ctx = Context::default().fst(a.clone());
        let new_history = History::default()
            .last_fn(Function::U(Rc::new(u)))
            .context(ctx)
            .push_input(a);
        Tensor::new(res, new_history)
    }
}
