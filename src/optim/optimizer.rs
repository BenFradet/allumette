use std::collections::HashMap;

use crate::tensor::tensor::Tensor;

pub trait Optimizer {
    fn zero_grad(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor>;
    fn update(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor>;
}
