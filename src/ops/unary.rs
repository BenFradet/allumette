use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
};

pub trait Unary<E: Element, BT: BackendType, B: Backend<E, BT>> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &B) -> B;
    // TODO: remove ctx
    fn backward(&self, ctx: &Context<B>, d: &B) -> B;

    fn tag(&self) -> &'static str;
}
