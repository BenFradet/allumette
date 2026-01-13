use crate::{
    backend::backend::Backend,
    tensor::Tensor,
};

pub trait Optimizer<'a, B: Backend> {
    fn zero(&mut self);
    fn step(&mut self, lr_tensor: Tensor<'a, B>);
}
