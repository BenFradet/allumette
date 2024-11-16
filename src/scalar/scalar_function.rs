use std::fmt::Debug;

use crate::ops::{binary_ops::Binary, unary_ops::Unary};

pub enum ScalarFunction {
    U(Box<dyn Unary>),
    B(Box<dyn Binary>),
}

// TODO: find a way to debug
impl Debug for ScalarFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(_) => write!(f, "Unary: ???"),
            Self::B(_) => write!(f, "Binary: ???"),
        }
    }
}
