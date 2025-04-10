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
        let mut input_shape = t.data.shape.data().to_vec();
        let input_first_dim = input_shape[0];
        let mut weights_shape = self.weights.data.shape.data().to_vec();
        if input_shape.last() != weights_shape.first() {
            None
        } else {
            weights_shape.insert(0, 1);
            let reshaped_weights = self.weights.clone().view(Shape::new(weights_shape)).unwrap();

            input_shape.push(1);
            let reshaped_input = t.view(Shape::new(input_shape)).unwrap();

            let batch_product = reshaped_weights * reshaped_input;
            let summed_product = batch_product.sum(Some(1)).contiguous();

            let output_shape = vec![input_first_dim, self.out_size];
            let reshaped_output = summed_product.view(Shape::new(output_shape)).unwrap();

            let final_output = reshaped_output + self.biases.clone().view(Shape::new(vec![1, self.out_size])).unwrap();
            Some(final_output)
        }
    }

    fn param(shape: Shape) -> Tensor {
        let t = Tensor::from_data(TensorData::rand(shape));
        (t - Tensor::scalar(0.5)) * Tensor::scalar(2.)
    }
}
