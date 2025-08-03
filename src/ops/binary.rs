use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
};

pub trait Binary<E: Element, BT: BackendType, B: Backend<E, BT>> {
    fn forward(&self, lhs: &B, rhs: &B) -> B;
    fn backward(&self, ctx: &Context<B>, d: &B) -> (B, B);
    fn tag(&self) -> &str;
}
