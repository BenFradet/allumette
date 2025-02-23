use crate::autodiff::history::History;

use super::tensor_data::TensorData;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: TensorData,
    pub history: History<TensorData>,
}

impl Tensor {
    pub fn new(data: TensorData, history: History<TensorData>) -> Self {
        Self { data, history }
    }
}
