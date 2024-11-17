use crate::{
    ops::{binary_ops::Binary, unary_ops::Unary},
    scalar::{scalar::Scalar, scalar_function::ScalarFunction, scalar_history::ScalarHistory},
};

use super::context::Context;

// TODO: abstract over scalar
pub struct Forward;
impl Forward {
    pub fn binary(b: impl Binary + 'static, lhs: Scalar, rhs: Scalar) -> Scalar {
        let res = b.forward(lhs.v, rhs.v);
        let ctx = Context::default()
            .push(lhs.v)
            .push(rhs.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::B(Box::new(b)))
            .context(ctx)
            .push_input(lhs)
            .push_input(rhs);
        Scalar::new(res).history(new_history)
    }

    pub fn unary(u: impl Unary + 'static, s: Scalar) -> Scalar {
        let res = u.forward(s.v);
        let ctx = Context::default().push(s.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::U(Box::new(u)))
            .context(ctx)
            .push_input(s);
        Scalar::new(res).history(new_history)
    }
}
