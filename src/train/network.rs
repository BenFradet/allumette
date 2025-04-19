use std::collections::HashMap;

use crate::tensor::tensor::Tensor;

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

    pub fn init(&self) -> HashMap<String, Tensor> {
        HashMap::from([
            (self.layer1.wkey(), self.layer1.weights.clone()),
            (self.layer1.bkey(), self.layer1.biases.clone()),
            (self.layer2.wkey(), self.layer2.weights.clone()),
            (self.layer2.bkey(), self.layer2.biases.clone()),
            (self.layer3.wkey(), self.layer3.weights.clone()),
            (self.layer3.bkey(), self.layer3.biases.clone()),
        ])
    }

    pub fn update(mut self, tensors: &HashMap<String, Tensor>) -> Self {
        let l1_wkey = self.layer1.wkey();
        let l1_bkey = self.layer1.bkey();
        self.layer1 = self
            .layer1
            .update_weights(tensors.get(&l1_wkey).unwrap().clone())
            .update_biases(tensors.get(&l1_bkey).unwrap().clone());
        let l2_wkey = self.layer2.wkey();
        let l2_bkey = self.layer2.bkey();
        self.layer2 = self
            .layer2
            .update_weights(tensors.get(&l2_wkey).unwrap().clone())
            .update_biases(tensors.get(&l2_bkey).unwrap().clone());
        let l3_wkey = self.layer3.wkey();
        let l3_bkey = self.layer3.bkey();
        self.layer3 = self
            .layer3
            .update_weights(tensors.get(&l3_wkey).unwrap().clone())
            .update_biases(tensors.get(&l3_bkey).unwrap().clone());
        self
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        let l1 = self.layer1.forward(x).relu();
        let l2 = self.layer2.forward(l1).relu();
        self.layer3.forward(l2).sigmoid()
    }
}
