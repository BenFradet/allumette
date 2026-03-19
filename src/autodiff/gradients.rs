use std::collections::HashMap;

use crate::{backend::backend::Backend, tensor::Tensor};

pub struct Gradients<'a, B: Backend>(pub HashMap<u64, Tensor<'a, B>>);

impl<'a, B: Backend> Gradients<'a, B> {
    pub fn wrt(&self, t: &Tensor<'a, B>) -> Option<&Tensor<'a, B>> {
        self.0.get(&t.id).and_then(|t| t.grad.as_deref())
    }
}
