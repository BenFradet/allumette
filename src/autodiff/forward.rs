use super::{
    scalar::Scalar,
    scalar_function::{Binary, ScalarFunction, Unary},
    scalar_history::ScalarHistory,
};

pub struct Forward;
impl Forward {
    pub fn binary(b: impl Binary + 'static, lhs: Scalar, rhs: Scalar) -> Scalar {
        let res = b.forward(&lhs.history.ctx, lhs.v, rhs.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::B(Box::new(b)))
            .push_input(lhs)
            .push_input(rhs);
        Scalar::new(res).history(new_history)
    }

    pub fn unary(u: impl Unary + 'static, s: Scalar) -> Scalar {
        let res = u.forward(&s.history.ctx, s.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::U(Box::new(u)))
            .push_input(s);
        Scalar::new(res).history(new_history)
    }
}
