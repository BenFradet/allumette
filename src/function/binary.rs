use crate::autodiff::context::Context;

pub trait Binary<A> {
    fn forward(&self, lhs: &A, rhs: &A) -> A;
    fn backward(&self, ctx: &Context<A>, d: &A) -> (A, A);
}
