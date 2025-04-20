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
        println!("l1 weights {:#?}", layer1.weights.data.data);
        println!("l1 biases {:#?}", layer1.biases.data.data);
        let layer2 = Layer::new("layer2", hidden_layer_size, hidden_layer_size);
        println!("l2 weights {:#?}", layer2.weights.data.data);
        println!("l2 biases {:#?}", layer2.biases.data.data);
        let layer3 = Layer::new("layer3", hidden_layer_size, 1);
        println!("l3 weights {:#?}", layer3.weights.data.data);
        println!("l3 biases {:#?}", layer3.biases.data.data);
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
        println!("network input {:#?}", x.data.data);
        println!("network input shape {:#?}", x.data.shape.data());
        let l1p = self.layer1.forward(x.clone());
        println!("network l1 before relu {:#?}", l1p.data.data);
        println!("network l1 shape before relu {:#?}", l1p.data.shape.data());
        let l1 = self.layer1.forward(x).relu();
        println!("network l1 {:#?}", l1.data.data);
        println!("network l1 shape {:#?}", l1.data.shape.data());
        let l2 = self.layer2.forward(l1).relu();
        println!("network l2 {:#?}", l2.data.data);
        println!("network l2 shape {:#?}", l2.data.shape.data());
        self.layer3.forward(l2).sigmoid()
    }
}
