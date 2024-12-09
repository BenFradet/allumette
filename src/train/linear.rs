use crate::{module::{module::Module, parameter::Parameter}, scalar::scalar::Scalar};

use rand::{thread_rng, Rng};

pub struct Linear {
    weights: Vec<Vec<Parameter>>,
    bias: Vec<Parameter>,
    out_size: usize,
    in_size: usize,
}

impl Linear {
    #[allow(clippy::needless_range_loop)]
    pub fn new(module: &mut Module, in_size: usize, out_size: usize) -> Self {
        let mut weights = vec![vec![]];
        let mut bias = vec![];
        let mut rng = thread_rng();

        for i in 0..in_size {
            weights[i] = vec![];
            for j in 0..out_size {
                let rand: f64 = rng.gen();
                let scalar = Scalar::new(2. * (rand - 0.5));
                let param = Parameter::new(format!("weight_{i}_{j}"), scalar);
                module.add_param(param.clone());
                weights[i][j] = param;
            }
        }

        for j in 0..out_size {
            let rand: f64 = rng.gen();
            let scalar = Scalar::new(2. * (rand - 0.5));
            let param = Parameter::new(format!("bias_{j}"), scalar);
            module.add_param(param.clone());
            bias[j] = param;
        }

        Self {
            weights,
            bias,
            out_size,
            in_size,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, inputs: Vec<Scalar>) -> impl Iterator<Item = Scalar> {
        let mut outputs = vec![];
        for j in 0..self.out_size {
            for i in 0..self.in_size {
                outputs[j] = &outputs[j] + &inputs[i] * &self.weights[i][j].scalar;
            }
            outputs[j] = &outputs[j] + &self.bias[j].scalar;
        }
        outputs.into_iter()
    }
}
