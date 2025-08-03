use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    tensor::Tensor,
};

pub trait Optimizer<E: Element, BT: BackendType, T: Backend<E, BT>> {
    fn zero(&mut self);
    fn step(&mut self, lr_tensor: Tensor<E, BT, T>);
}
