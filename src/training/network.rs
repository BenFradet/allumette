use crate::{
    autodiff::gradients::Gradients, backend::backend::Backend, optim::optimizer::Optimizer,
    tensor::Tensor,
};

use super::layer::Layer;

pub struct Network<'a, B: Backend> {
    layer1: Layer<'a, B>,
    layer2: Layer<'a, B>,
    layer3: Layer<'a, B>,
}

impl<'a, B: Backend> Network<'a, B> {
    pub fn new(hidden_layer_size: usize) -> Self {
        let layer1 = Layer::new("layer1", 2, hidden_layer_size);
        let layer2 = Layer::new("layer2", hidden_layer_size, hidden_layer_size);
        let layer3 = Layer::new("layer3", hidden_layer_size, 1);
        Self {
            layer1,
            layer2,
            layer3,
        }
    }

    pub fn step(&mut self, optimizer: &impl Optimizer<'a, B>, gradients: &Gradients<'a, B>) {
        optimizer.update(&mut self.layer1.weights, gradients);
        optimizer.update(&mut self.layer1.biases, gradients);
        optimizer.update(&mut self.layer2.weights, gradients);
        optimizer.update(&mut self.layer2.biases, gradients);
        optimizer.update(&mut self.layer3.weights, gradients);
        optimizer.update(&mut self.layer3.biases, gradients);
    }

    pub fn forward(&self, x: Tensor<'a, B>) -> Tensor<'a, B> {
        let l1 = self.layer1.forward(x).relu();
        let l2 = self.layer2.forward(l1).relu();
        self.layer3.forward(l2).sig()
    }
}
