use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

pub trait ScalarOptimizer {
    fn zero(&self, scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar>;
    fn step(&self, scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar>;
}
