use std::collections::HashMap;

use crate::tensor::tensor::Tensor;

use super::layer::Layer;

pub struct Network {
    layer1: Layer,
    layer2: Layer,
    layer3: Layer,
}

impl Network {
    pub fn new(hidden_layer_size: usize) -> Self {
        let layer1 = Layer::new("layer1".to_owned(), 2, hidden_layer_size);
        let layer2 = Layer::new("layer2".to_owned(), hidden_layer_size, hidden_layer_size);
        let layer3 = Layer::new("layer3".to_owned(), hidden_layer_size, 1);
        Self {
            layer1,
            layer2,
            layer3,
        }
    }

    pub fn init(&self) -> HashMap<String, Tensor> {
        HashMap::from([
            ("layer1_weights".to_owned(), self.layer1.weights.clone()),
            ("layer1_biases".to_owned(), self.layer1.biases.clone()),
            ("layer2_weights".to_owned(), self.layer2.weights.clone()),
            ("layer2_biases".to_owned(), self.layer2.biases.clone()),
            ("layer3_weights".to_owned(), self.layer3.weights.clone()),
            ("layer3_biases".to_owned(), self.layer3.biases.clone()),
        ])
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        let l1 = self.layer1.forward(x);
        let l2 = self.layer2.forward(l1.relu());
        let l3 = self.layer3.forward(l2.relu());
        l3.sigmoid()
    }
}
