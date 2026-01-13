use crate::{
    autodiff::context::Context,
    backend::backend::Backend,
};

pub trait Binary<'a, B: Backend> {
    fn forward(&self, lhs: &B::Storage<'a>, rhs: &B::Storage<'a>) -> B::Storage<'a>;
    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> (B::Storage<'a>, B::Storage<'a>);
    fn tag(&self) -> &'static str;
}
