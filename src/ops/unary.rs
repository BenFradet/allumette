use crate::{
    autodiff::context::Context,
    backend::backend::Backend,
};

pub trait Unary<'a, B: Backend> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a>;
    // TODO: remove ctx
    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a>;

    fn tag(&self) -> &'static str;
}
