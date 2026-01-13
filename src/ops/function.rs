use std::{fmt::Debug, rc::Rc};

use crate::{
    backend::{backend::{Backend, TensorBackend}, backend_type::BackendType}, data::tensor_data::TensorData, math::element::Element
};

use super::{binary::Binary, unary::Unary};

#[derive(Clone)]
pub enum Function<'a, B: Backend> {
    U(Rc<dyn Unary<'a, B>>),
    B(Rc<dyn Binary<'a, B>>),
}

impl<'a, B: Backend> Debug for Function<'a, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(u) => write!(f, "Unary: {}", u.tag()),
            Self::B(b) => write!(f, "Binary: {}", b.tag()),
        }
    }
}
