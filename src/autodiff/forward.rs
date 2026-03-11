use std::rc::Rc;

use crate::{
    autodiff::trace::Trace,
    backend::backend::Backend,
    fns::{binary::Binary, function::Function, unary::Unary},
    tensor::Tensor,
};

pub struct Forward;

impl Forward {
    pub fn binary<'a, B: Backend>(
        b: impl Binary<'a, B> + 'static,
        lhs: Tensor<'a, B>,
        rhs: Tensor<'a, B>,
    ) -> Tensor<'a, B> {
        let res = b.forward(&lhs.data, &rhs.data);
        let new_trace = if lhs.is_constant && rhs.is_constant {
            Trace::default()
        } else {
            Trace::default()
                .last_fn(Function::B(Rc::new(b)))
                .push_input(lhs)
                .push_input(rhs)
        };
        Tensor::new(res, new_trace)
    }

    pub fn unary<'a, B: Backend>(
        u: impl Unary<'a, B> + 'static,
        a: Tensor<'a, B>,
    ) -> Tensor<'a, B> {
        let res = u.forward(&a.data);
        let new_trace = if a.is_constant {
            Trace::default()
        } else {
            Trace::default()
                .last_fn(Function::U(Rc::new(u)))
                .push_input(a)
        };
        Tensor::new(res, new_trace)
    }
}
