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

    pub fn forward(&self, x: f64, y: f64) -> f64 {
        //middle = self.layer1.forward(vec![x, y]).map(|m| m.rel)
        0.
    }
}