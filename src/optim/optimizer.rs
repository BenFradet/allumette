use crate::{autodiff::gradients::Gradients, backend::backend::Backend, tensor::Tensor};

pub trait Optimizer<'a, B: Backend> {
    fn update(&self, param: &mut Tensor<'a, B>, gradients: &Gradients<'a, B>);
}
