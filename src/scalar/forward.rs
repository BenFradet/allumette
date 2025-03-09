use std::rc::Rc;

use crate::{
    autodiff::context::Context,
    function::{binary::Binary, function::Function, unary::Unary},
    scalar::{scalar::Scalar, scalar_history::ScalarHistory},
};

pub struct Forward;
impl Forward {
    pub fn binary(b: impl Binary<f64> + 'static, lhs: &Scalar, rhs: &Scalar) -> Scalar {
        let res = b.forward(&lhs.v, &rhs.v);
        let ctx = Context::default().fst(lhs.v).snd(rhs.v);
        let new_history = ScalarHistory::default()
            .last_fn(Function::B(Rc::new(b)))
            .context(ctx)
            .push_input(lhs.clone())
            .push_input(rhs.clone());
        Scalar::new(res).history(new_history)
    }

    pub fn unary(u: impl Unary<f64> + 'static, s: &Scalar) -> Scalar {
        let res = u.forward(s.v);
        let ctx = Context::default().fst(s.v);
        let new_history = ScalarHistory::default()
            .last_fn(Function::U(Rc::new(u)))
            .context(ctx)
            .push_input(s.clone());
        Scalar::new(res).history(new_history)
    }
}
