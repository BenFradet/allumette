use std::{fmt::Debug, rc::Rc};

use super::{binary::Binary, unary::Unary};

// TODO: find a way to partial eq
#[derive(Clone)]
pub enum Function<A, B> {
    U(Rc<dyn Unary<A>>),
    B(Rc<dyn Binary<A, B>>),
}

// TODO: find a way to debug
impl<A, B> Debug for Function<A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(_) => write!(f, "Unary: ???"),
            Self::B(_) => write!(f, "Binary: ???"),
        }
    }
}
