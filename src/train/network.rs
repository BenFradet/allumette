use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

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

    pub fn init(&self) -> HashMap<String, Scalar> {
        let mut res = self.layer1.init_weights();
        let l1_b = self.layer1.init_bias();
        res.extend(l1_b);
        let l2_w = self.layer2.init_weights();
        res.extend(l2_w);
        let l2_b = self.layer2.init_bias();
        res.extend(l2_b);
        let l3_w = self.layer3.init_weights();
        res.extend(l3_w);
        let l3_b = self.layer3.init_bias();
        res.extend(l3_b);
        res
    }

    pub fn forward(
        &self,
        x1: Scalar,
        x2: Scalar,
        scalars: &HashMap<String, Scalar>,
    ) -> Option<Scalar> {
        let l1 = self.layer1.forward(vec![x1, x2], scalars).map(|s| s.relu());
        let l2 = self.layer2.forward(l1.collect(), scalars).map(|s| s.relu());
        self.layer3
            .forward(l2.collect(), scalars)
            .next()
            .map(|s| s.sig())
    }
}
