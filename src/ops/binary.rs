use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
};

pub trait Binary<B: Backend> {
    fn forward(&self, lhs: &B::Storage, rhs: &B::Storage) -> B::Storage;
    fn backward(&self, ctx: &Context<B::Storage>, d: &B::Storage) -> (B::Storage, B::Storage);
    fn tag(&self) -> &'static str;
}
