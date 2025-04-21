use std::collections::HashMap;

use crate::{
    optim::optimizer::Optimizer,
    tensor::{tensor::Tensor, tensor_history::TensorHistory},
};

use super::layer::Layer;

pub struct Network<'a> {
    layer1: Layer<'a>,
    layer2: Layer<'a>,
    layer3: Layer<'a>,
}

impl Network<'_> {
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

    pub fn update(&mut self, tensors: &HashMap<String, Tensor>) {
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

    pub fn forward(&self, x: Tensor) -> Tensor {
        println!("network input {:#?}", x.data.data);
        println!("network input shape {:#?}", x.data.shape.data());
        let l1 = self.layer1.forward(x).relu();
        println!("network l1 {:#?}", l1.data.data);
        println!("network l1 shape {:#?}", l1.data.shape.data());
        let l2 = self.layer2.forward(l1).relu();
        println!("network l2 {:#?}", l2.data.data);
        println!("network l2 shape {:#?}", l2.data.shape.data());
        let l3 = self.layer3.forward(l2).sigmoid();
        println!("network l3 {:#?}", l3.data.data);
        println!("network l3 shape {:#?}", l3.data.shape.data());
        l3
    }
}

impl Optimizer for Network<'_> {
    fn zero(&mut self) {
        self.layer1.weights.grad = None;
        self.layer2.weights.grad = None;
        self.layer3.weights.grad = None;
        self.layer1.biases.grad = None;
        self.layer2.biases.grad = None;
        self.layer3.biases.grad = None;
    }

    fn step(&mut self, lr_tensor: Tensor) {
        if let Some(grad) = &self.layer1.weights.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer1.weights.id, self.layer1.weights.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer1.weights = (self.layer1.weights.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer1.weights.id.clone());
            println!("to {:#?}", self.layer1.weights.data.data);
        }
        if let Some(grad) = &self.layer2.weights.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer2.weights.id, self.layer2.weights.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer2.weights = (self.layer2.weights.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer2.weights.id.clone());
            println!("to {:#?}", self.layer2.weights.data.data);
        }
        if let Some(grad) = &self.layer3.weights.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer3.weights.id, self.layer3.weights.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer3.weights = (self.layer3.weights.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer3.weights.id.clone());
            println!("to {:#?}", self.layer3.weights.data.data);
        }
        if let Some(grad) = &self.layer1.biases.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer1.biases.id, self.layer1.biases.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer1.biases = (self.layer1.biases.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer1.biases.id.clone());
            println!("to {:#?}", self.layer1.biases.data.data);
        }
        if let Some(grad) = &self.layer2.biases.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer2.biases.id, self.layer2.biases.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer2.biases = (self.layer2.biases.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer2.biases.id.clone());
            println!("to {:#?}", self.layer2.biases.data.data);
        }
        if let Some(grad) = &self.layer3.biases.grad {
            println!(
                "updating {:?} {:#?} with {:#?}",
                self.layer3.biases.id, self.layer3.biases.data.data, grad.data.data
            );
            let update = lr_tensor.clone() * *grad.clone();
            self.layer3.biases = (self.layer3.biases.clone() - update)
                .history(TensorHistory::default())
                .id(self.layer3.biases.id.clone());
            println!("to {:#?}", self.layer3.biases.data.data);
        }
    }
}
