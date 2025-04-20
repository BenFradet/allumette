use crate::tensor::tensor::Tensor;

pub trait Optimizer {
    fn zero(&mut self);
    fn step(&mut self, lr_tensor: Tensor);
}
