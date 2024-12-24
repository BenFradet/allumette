use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

use rand::{thread_rng, Rng};

pub struct Layer {
    pub name: String,
    pub in_size: usize,
    pub out_size: usize,
}

impl Layer {
    pub fn new(name: String, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
        }
    }

    // TODO: use module
    #[allow(clippy::needless_range_loop)]
    pub fn weights(&self) -> HashMap<String, Scalar> {
        let mut weights = HashMap::new();
        let mut rng = thread_rng();

        for i in 0..self.in_size {
            for j in 0..self.out_size {
                let rand: f64 = rng.gen();
                let scalar_id = self.weight_key(i, j);
                let scalar = Scalar::new(2. * (rand - 0.5)).id(scalar_id.clone());
                weights.insert(scalar_id, scalar);
            }
        }
        weights
    }

    #[allow(clippy::needless_range_loop)]
    pub fn biases(&self) -> HashMap<String, Scalar> {
        let mut bias = HashMap::new();
        let mut rng = thread_rng();

        for j in 0..self.out_size {
            let rand: f64 = rng.gen();
            let scalar_id = self.bias_key(j);
            let scalar = Scalar::new(2. * (rand - 0.5)).id(scalar_id.clone());
            bias.insert(scalar_id, scalar);
        }
        bias
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward(
        &self,
        inputs: Vec<Scalar>,
        scalars: &HashMap<String, Scalar>,
    ) -> impl Iterator<Item = Scalar> {
        let mut outputs = vec![Scalar::new(0.); self.out_size];
        let z = Scalar::new(0.);
        for j in 0..self.out_size {
            for i in 0..self.in_size {
                let wk = self.weight_key(i, j);
                let w = scalars.get(&wk).unwrap_or(&z);
                outputs[j] = &outputs[j] + &inputs[i] * &w;
            }
            let bk = self.bias_key(j);
            let b = scalars.get(&bk).unwrap_or(&z);
            outputs[j] = &outputs[j] + b;
        }
        outputs.into_iter()
    }

    fn weight_key(&self, i: usize, j: usize) -> String {
        format!("{}_weight_{i}_{j}", self.name)
    }

    fn bias_key(&self, j: usize) -> String {
        format!("{}_bias_{j}", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biases_test() -> () {
        let layer = Layer::new("layer".to_owned(), 2, 3);
        let res = layer.biases();
        assert!(res.iter().all(|(k, v)| k.starts_with("layer_bias") && v.v <= 1. && v.v >= -1.));
    }

    #[test]
    fn weights_test() -> () {
        let layer = Layer::new("layer".to_owned(), 2, 3);
        let res = layer.weights();
        assert!(res.iter().all(|(k, v)| k.starts_with("layer_weight") && v.v <= 1. && v.v >= -1.));
    }

    #[test]
    fn weight_key_test() -> () {
        let layer = Layer::new("layer".to_owned(), 1, 1);
        assert_eq!("layer_weight_0_0", layer.weight_key(0, 0));
    }

    #[test]
    fn bias_key_test() -> () {
        let layer = Layer::new("layer".to_owned(), 1, 1);
        assert_eq!("layer_bias_0", layer.bias_key(0));
    }
}
