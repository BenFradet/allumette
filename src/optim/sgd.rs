use crate::{
    autodiff::{gradients::Gradients, trace::Trace},
    backend::backend::Backend,
    optim::optimizer::Optimizer,
    tensor::Tensor,
};

pub struct SGD<'a, B: Backend> {
    lr: Tensor<'a, B>,
}

impl<'a, B: Backend> SGD<'a, B> {
    pub fn new(learning_rate: B::Element) -> Self {
        Self {
            lr: Tensor::from_scalar(learning_rate),
        }
    }
}

impl<'a, B: Backend> Optimizer<'a, B> for SGD<'a, B> {
    fn update(&self, param: &mut crate::tensor::Tensor<'a, B>, gradients: &Gradients<'a, B>) {
        if let Some(grad) = gradients.wrt(param) {
            *param = (param.clone() - self.lr.clone() * grad.clone())
                .trace(Trace::default())
                .id(param.id.clone());
        }
    }
}
