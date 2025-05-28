use crate::{
    backend::{backend::TensorBackend, backend_type::TensorBackendType},
    tensor::Tensor,
};

pub trait Optimizer<BT: TensorBackendType, T: TensorBackend<BT>> {
    fn zero(&mut self);
    fn step(&mut self, lr_tensor: Tensor<BT, T>);
}
