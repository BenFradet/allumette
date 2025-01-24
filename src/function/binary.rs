use crate::autodiff::context::Context;

pub trait Binary<A, B> {
    fn forward(&self, a: A, b: B) -> A;
    fn backward(&self, ctx: &Context<A, B>, d: A) -> (A, A);
}
