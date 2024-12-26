use std::sync::Arc;

use super::tensor_data::TensorData;

#[derive(Clone)]
pub struct Tensor<const N: usize> {
    data: Arc<TensorData<N>>,
}
