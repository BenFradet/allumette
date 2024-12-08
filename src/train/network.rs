use crate::scalar::scalar::Scalar;

use super::linear::Linear;

pub struct Network {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl Network {
    pub fn new(hidden_layer_size: usize) -> Self {
        Self {
            layer1: Linear::new(2, hidden_layer_size),
            layer2: Linear::new(hidden_layer_size, hidden_layer_size),
            layer3: Linear::new(hidden_layer_size, 1),
        }
    }

    pub fn forward(&self, x: Scalar) -> Option<Scalar> {
        let l1 = self.layer1.forward(vec![x]).map(|s| s.relu());
        let l2 = self.layer2.forward(l1.collect()).map(|s| s.relu());
        let l3 = self.layer3.forward(l2.collect()).next().map(|s| s.sig());
        l3
    }
}