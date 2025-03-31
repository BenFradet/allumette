use crate::autodiff::context::Context;

pub trait Unary<A> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &A) -> A;
    // TODO: remove ctx
    fn backward(&self, ctx: &Context<A>, d: &A) -> A;

    fn tag(&self) -> &str;
}
