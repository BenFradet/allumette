use std::collections::HashMap;

use crate::tensor::{
    shaping::shape::Shape, tensor::Tensor, tensor_data::TensorData, tensor_history::TensorHistory,
};

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

    pub fn forward(&self, t: Tensor, tensors: &HashMap<String, Tensor>) -> Tensor {
        let mut input_shape = t.data.shape.data().to_vec();
        let input_first_dim = input_shape[0];

        let z = Tensor::scalar(0.);

        let w_key = self.weights_key();
        let weights = tensors.get(&w_key).unwrap_or(&z);
        let mut weights_shape = weights.data.shape.data().to_vec();

        // input size must match weight size
        assert!(
            input_shape.last() == weights_shape.first(),
            "input size does not match weight size"
        );

        // reshape weights to prepare for matrix multiplication by adding a batch dimension
        weights_shape.insert(0, 1);
        let reshaped_weights = weights.clone().view(&Shape::new(weights_shape)).unwrap();

        // reshape input tensor by adding an output dimension
        input_shape.push(1);
        let reshaped_input = t.view(&Shape::new(input_shape)).unwrap();

        // perform element-wise multiplication and then sum across the input dimension
        // to perform a dot product equivalent for each sample in the batch
        let batch_product = reshaped_weights * reshaped_input;
        let summed_product = batch_product.sum(Some(1)).contiguous();

        // reshape the summed product to match the output dimension
        // number of samples by output size
        let output_shape = vec![input_first_dim, self.out_size];
        let reshaped_output = summed_product.view(&Shape::new(output_shape)).unwrap();

        // add bias to each output in the batch
        // bias is reshaped to match the batch output shape (1, out_size) for broadcasting
        let b_key = self.biases_key();
        let biases = tensors.get(&b_key).unwrap_or(&z);
        reshaped_output
            + biases
                .clone()
                .view(&Shape::new(vec![1, self.out_size]))
                .unwrap()
    }

    pub fn weights(&self) -> Tensor {
        let id = self.weights_key();
        Self::param(Shape::new(vec![self.in_size, self.out_size])).id(id)
    }

    pub fn biases(&self) -> Tensor {
        let id = self.biases_key();
        Self::param(Shape::new(vec![self.out_size])).id(id)
    }

    fn param(shape: Shape) -> Tensor {
        let t = Tensor::from_data(TensorData::rand(shape));
        ((t - Tensor::scalar(0.5)) * Tensor::scalar(2.)).history(TensorHistory::default())
    }

    fn weights_key(&self) -> String {
        format!("{}_weights", self.name)
    }

    fn biases_key(&self) -> String {
        format!("{}_biases", self.name)
    }
}
