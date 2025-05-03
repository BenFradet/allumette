use std::{fmt::Debug, rc::Rc};

use crate::backend::{backend::Backend, backend_type::BackendType};

use super::{binary::Binary, unary::Unary};

// TODO: find a way to partial eq
#[derive(Clone)]
pub enum Function<BT: BackendType, B: Backend<BT>> {
    U(Rc<dyn Unary<BT, B>>),
    B(Rc<dyn Binary<BT, B>>),
}

impl<BT: BackendType, B: Backend<BT>> Debug for Function<BT, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(u) => write!(f, "Unary: {}", u.tag()),
            Self::B(b) => write!(f, "Binary: {}", b.tag()),
        }
    }
}
