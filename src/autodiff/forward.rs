use std::{marker::PhantomData, rc::Rc};

use crate::{
    function::{binary::Binary, function::Function, unary::Unary},
    scalar::{scalar::Scalar, scalar_history::ScalarHistory},
};

use super::{
    context::Context,
    history::{HasHistory, History},
};

// TODO: abstract over scalar
pub struct Forward;
impl Forward {
    pub fn binary(b: impl Binary<f64> + 'static, lhs: &Scalar, rhs: &Scalar) -> Scalar {
        let res = b.forward(lhs.v, rhs.v);
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

pub struct Forward2<A> {
    phantom: PhantomData<A>,
}

impl<A: Clone + Default + HasHistory> Forward2<A> {
    pub fn binary(b: impl Binary<A> + 'static, lhs: A, rhs: A) -> A {
        let res = b.forward(lhs.clone(), rhs.clone());
        let ctx = Context::default().fst(lhs.clone()).snd(rhs.clone());
        let new_history = History::default()
            .last_fn(Function::B(Rc::new(b)))
            .context(ctx)
            .push_input(lhs)
            .push_input(rhs);
        res.history(new_history)
    }

    pub fn unary(u: impl Unary<A> + 'static, a: A) -> A {
        let res = u.forward(a.clone());
        let ctx = Context::default().fst(a.clone());
        let new_history = History::default()
            .last_fn(Function::U(Rc::new(u)))
            .context(ctx)
            .push_input(a);
        res.history(new_history)
    }
}
