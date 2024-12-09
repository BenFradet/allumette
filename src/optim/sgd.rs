use crate::{module::module::Module, scalar::scalar::Scalar};

use super::optimizer::Optimizer;

pub struct SGD<'a> {
    module: &'a mut Module,
    lr: f64,
}

impl<'a> SGD<'a> {
    pub fn new(module: &'a mut Module, lr: f64) -> Self {
        Self { module, lr }
    }

}

impl<'a> Optimizer for SGD<'a> {
    fn zero(&mut self) -> () {
        for (_, p) in &mut self.module.parameters {
            p.scalar.derivative = None;
        }
    }

    fn step(&mut self) -> () {
        for (_, p) in &mut self.module.parameters {
            if let Some(d) = p.scalar.derivative {
                p.scalar = Scalar::new(p.scalar.v - self.lr * d);
            }
            p.scalar.derivative = None;
        }
    }
}
