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

    pub fn forward(&self, x: Tensor) -> Option<Tensor> {
        self.layer1.forward(x)
            .and_then(|t1| self.layer2.forward(t1.relu()))
            .and_then(|t2| self.layer3.forward(t2.relu()))
            .map(|t3| t3.sigmoid())
    }
}
