use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

use super::optimizer::Optimizer;

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn zero(&self, mut scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar> {
        for s in scalars.values_mut() {
            s.derivative = None;
        }
        scalars
    }

    fn step(&self, mut scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar> {
        for s in scalars.values_mut() {
            if let Some(d) = s.derivative {
                *s = Scalar::new(s.v - self.lr * d).id(s.id.clone());
            }
            s.derivative = None;
        }
        scalars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_test() -> () {
        let s = Scalar::new(1.);
        let map = HashMap::from([(s.id.clone(), s)]);
        let sgd = SGD::new(0.5);
        let res = sgd.zero(map);
        assert!(res.iter().all(|kv| kv.1.derivative == None));
    }

    #[test]
    fn step_test() -> () {
        let s = Scalar::new(0.5).derivative(Some(0.75));
        let s_id = s.id.clone();
        let map = HashMap::from([(s_id.clone(), s)]);
        let sgd = SGD::new(0.25);
        let res = sgd.step(map);
        assert!(res
            .iter()
            .all(|kv| kv.1.derivative == None && kv.1.v == 0.3125 && kv.1.id == s_id));
    }
}
