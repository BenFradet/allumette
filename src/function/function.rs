use std::{fmt::Debug, rc::Rc};

use super::{binary::Binary, unary::Unary};

// TODO: find a way to partial eq
#[derive(Clone)]
pub enum Function<A> {
    U(Rc<dyn Unary<A>>),
    B(Rc<dyn Binary<A>>),
}

// TODO: find a way to debug
impl<A> Debug for Function<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(u) => write!(f, "Unary: {}", u.tag()),
            Self::B(_) => write!(f, "Binary: ???"),
        }
    }
}
