use std::collections::HashMap;

use crate::{scalar::scalar::Scalar, tensor::tensor::Tensor};

use super::scalar_optimizer::ScalarOptimizer;

pub struct SGD {
    lr: f64,
    lr_tensor: Tensor,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            lr_tensor: Tensor::scalar(lr),
        }
    }
}

//impl Optimizer for SGD {
//    fn zero_grad(&self, mut tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
//        for t in tensors.values_mut() {
//            t.grad = None;
//        }
//        tensors
//    }
//
//    fn update(&self, mut tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
//        for t in tensors.values_mut() {
//            if let Some(grad) = &t.grad {
//                println!("updating {:?} {:#?}", t.id, t.data.data);
//                let update = self.lr_tensor.clone() * *grad.clone();
//                *t = (t.clone() - update)
//                    .history(TensorHistory::default())
//                    .id(t.id.clone());
//                println!("to {:#?}", t.data.data);
//            }
//            t.grad = None;
//        }
//        tensors
//    }
//}

impl ScalarOptimizer for SGD {
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
