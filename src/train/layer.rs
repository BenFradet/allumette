use crate::tensor::{shaping::shape::Shape, tensor::Tensor, tensor_data::TensorData};

pub struct Layer {
    pub name: String,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Tensor,
    pub biases: Tensor,
}

impl Layer {
    pub fn new(name: String, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
            weights: Self::param(Shape::new(vec![in_size, out_size])),
            biases: Self::param(Shape::new(vec![out_size])),
        }
    }

    pub fn forward(&self, t: Tensor) -> Option<Tensor> {
        if t.data.shape.data().last() != self.weights.data.shape.data().first() {
            None
        } else {
            todo!()
        }
    }

    fn param(shape: Shape) -> Tensor {
        let t = Tensor::from_data(TensorData::rand(shape));
        (t - Tensor::scalar(0.5)) * Tensor::scalar(2.)
    }
}
