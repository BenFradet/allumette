use crate::tensor::tensor::Tensor;

pub struct Layer {
    pub name: String,
    pub in_size: usize,
    pub out_size: usize,
}

impl Layer {
    pub fn new(name: String, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
        }
    }

    pub fn forward(&self, _t: Tensor) -> Tensor {
        todo!()
    }
}
