use crate::{
    autodiff::gradients::Gradients, backend::backend::Backend, optim::optimizer::Optimizer,
    tensor::Tensor,
};

use super::layer::Layer;

pub struct Network<'a, B: Backend> {
    input_layer: Layer<'a, B>,
    hidden_layer: Layer<'a, B>,
    output_layer: Layer<'a, B>,
}

impl<'a, B: Backend> Network<'a, B> {
    pub fn new(hidden_layer_size: usize) -> Self {
        let input = Layer::new(2, hidden_layer_size);
        let hidden = Layer::new(hidden_layer_size, hidden_layer_size);
        let output = Layer::new(hidden_layer_size, 1);
        Self {
            input_layer: input,
            hidden_layer: hidden,
            output_layer: output,
        }
    }

    pub fn step(&mut self, optimizer: &impl Optimizer<'a, B>, gradients: &Gradients<'a, B>) {
        optimizer.update(&mut self.input_layer.weights, gradients);
        optimizer.update(&mut self.input_layer.biases, gradients);
        optimizer.update(&mut self.hidden_layer.weights, gradients);
        optimizer.update(&mut self.hidden_layer.biases, gradients);
        optimizer.update(&mut self.output_layer.weights, gradients);
        optimizer.update(&mut self.output_layer.biases, gradients);
    }

    pub fn forward(&self, x: Tensor<'a, B>) -> Tensor<'a, B> {
        let l1 = self.input_layer.forward(x).relu();
        let l2 = self.hidden_layer.forward(l1).relu();
        self.output_layer.forward(l2).sig()
    }
}
