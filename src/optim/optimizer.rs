use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    tensor::Tensor,
};

pub trait Optimizer<BT: BackendType, T: Backend<BT>> {
    fn zero(&mut self);
    fn step(&mut self, lr_tensor: Tensor<BT, T>);
}
