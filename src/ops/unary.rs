use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
};

pub trait Unary<B: Backend> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &B::Storage) -> B::Storage;
    // TODO: remove ctx
    fn backward(&self, ctx: &Context<B::Storage>, d: &B::Storage) -> B::Storage;

    fn tag(&self) -> &'static str;
}
