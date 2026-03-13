use std::collections::HashMap;

use crate::{backend::backend::Backend, tensor::Tensor};

pub struct Gradients<'a, B: Backend>(pub HashMap<String, Tensor<'a, B>>);

impl<'a, B: Backend> Gradients<'a, B> {
    pub fn wrt(&self, t: &Tensor<'a, B>) -> &Tensor<'a, B> {
        self.0
            .get(&t.id)
            .expect("tensor not found in gradients")
            .grad
            .as_ref()
            .expect("tensor has no grad")
    }
}
