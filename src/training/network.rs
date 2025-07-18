use std::collections::HashMap;

use crate::{
    autodiff::history::History,
    backend::{backend::Backend, backend_type::BackendType},
    optim::optimizer::Optimizer,
    tensor::Tensor,
};

use super::layer::Layer;

pub struct Network<'a, BT: BackendType, T: Backend<BT>> {
    layer1: Layer<'a, BT, T>,
    layer2: Layer<'a, BT, T>,
    layer3: Layer<'a, BT, T>,
}

impl<BT: BackendType, T: Backend<BT>> Network<'_, BT, T> {
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

    pub fn update(&mut self, tensors: &HashMap<String, Tensor<BT, T>>) {
        let l1_wkey = self.layer1.wkey();
        let l1_bkey = self.layer1.bkey();
        self.layer1.weights = tensors.get(&l1_wkey).unwrap().clone();
        self.layer1.biases = tensors.get(&l1_bkey).unwrap().clone();
        let l2_wkey = self.layer2.wkey();
        let l2_bkey = self.layer2.bkey();
        self.layer2.weights = tensors.get(&l2_wkey).unwrap().clone();
        self.layer2.biases = tensors.get(&l2_bkey).unwrap().clone();
        let l3_wkey = self.layer3.wkey();
        let l3_bkey = self.layer3.bkey();
        self.layer3.weights = tensors.get(&l3_wkey).unwrap().clone();
        self.layer3.biases = tensors.get(&l3_bkey).unwrap().clone();
    }

    pub fn forward(&self, x: Tensor<BT, T>) -> Tensor<BT, T> {
        let l1 = self.layer1.forward(x).relu();
        let l2 = self.layer2.forward(l1).relu();
        self.layer3.forward(l2).sigmoid()
    }
}

impl<BT: BackendType, T: Backend<BT>> Optimizer<BT, T> for Network<'_, BT, T> {
    fn zero(&mut self) {
        self.layer1.weights.grad = None;
        self.layer2.weights.grad = None;
        self.layer3.weights.grad = None;
        self.layer1.biases.grad = None;
        self.layer2.biases.grad = None;
        self.layer3.biases.grad = None;
    }

    fn step(&mut self, lr_tensor: Tensor<BT, T>) {
        if let Some(grad) = &self.layer1.weights.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer1.weights = (self.layer1.weights.clone() - update)
                .history(History::default())
                .id(self.layer1.weights.id.clone());
        }
        if let Some(grad) = &self.layer1.biases.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer1.biases = (self.layer1.biases.clone() - update)
                .history(History::default())
                .id(self.layer1.biases.id.clone());
        }
        if let Some(grad) = &self.layer2.weights.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer2.weights = (self.layer2.weights.clone() - update)
                .history(History::default())
                .id(self.layer2.weights.id.clone());
        }
        if let Some(grad) = &self.layer2.biases.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer2.biases = (self.layer2.biases.clone() - update)
                .history(History::default())
                .id(self.layer2.biases.id.clone());
        }
        if let Some(grad) = &self.layer3.weights.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer3.weights = (self.layer3.weights.clone() - update)
                .history(History::default())
                .id(self.layer3.weights.id.clone());
        }
        if let Some(grad) = &self.layer3.biases.grad {
            let update = lr_tensor.clone() * *grad.clone();
            self.layer3.biases = (self.layer3.biases.clone() - update)
                .history(History::default())
                .id(self.layer3.biases.id.clone());
        }
    }
}
