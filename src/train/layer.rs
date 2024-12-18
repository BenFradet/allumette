use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

use rand::{thread_rng, Rng};

pub struct Layer {
    name: String,
    in_size: usize,
    out_size: usize,
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
    pub fn bias(&self) -> HashMap<String, Scalar> {
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
