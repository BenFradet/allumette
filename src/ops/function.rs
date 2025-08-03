use std::{fmt::Debug, rc::Rc};

use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
};

use super::{binary::Binary, unary::Unary};

#[derive(Clone)]
pub enum Function<E: Element, BT: BackendType, B: Backend<E, BT>> {
    U(Rc<dyn Unary<E, BT, B>>),
    B(Rc<dyn Binary<E, BT, B>>),
}

impl<E: Element, BT: BackendType, B: Backend<E, BT>> Debug for Function<E, BT, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(u) => write!(f, "Unary: {}", u.tag()),
            Self::B(b) => write!(f, "Binary: {}", b.tag()),
        }
    }
}
