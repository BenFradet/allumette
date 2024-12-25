use std::sync::Arc;

use super::tensor_data::TensorData;

#[derive(Clone)]
pub struct Tensor {
    data: Arc<TensorData>,
}