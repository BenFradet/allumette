use crate::{module::module::Module, scalar::scalar::Scalar};

use super::linear::Linear;

pub struct Network {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl Network {
    pub fn new(module: &mut Module, hidden_layer_size: usize) -> Self {
        let layer1 = Linear::new(module, 2, hidden_layer_size);
        let layer2 = Linear::new(module, hidden_layer_size, hidden_layer_size);
        let layer3 = Linear::new(module, hidden_layer_size, 1);
        Self {
            layer1,
            layer2,
            layer3,
        }
    }

    pub fn forward(&self, x1: Scalar, x2: Scalar) -> Option<Scalar> {
        let l1 = self.layer1.forward(vec![x1, x2]).map(|s| s.relu());
        let l2 = self.layer2.forward(l1.collect()).map(|s| s.relu());
        self.layer3.forward(l2.collect()).next().map(|s| s.sig())
    }
}