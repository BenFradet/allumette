use std::{fmt::Debug, rc::Rc};

use crate::ops::{binary_ops::Binary, unary_ops::Unary};

#[derive(Clone)]
pub enum ScalarFunction {
    U(Rc<dyn Unary>),
    B(Rc<dyn Binary>),
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
