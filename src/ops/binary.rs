use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
};

pub trait Binary<BT: BackendType, B: Backend<BT>> {
    fn forward(&self, lhs: &B, rhs: &B) -> B;
    fn backward(&self, ctx: &Context<B>, d: &B) -> (B, B);
    fn tag(&self) -> &str;
}
