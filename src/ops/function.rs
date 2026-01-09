use std::{fmt::Debug, rc::Rc};

use crate::{
    backend::{backend::{Backend, TensorBackend}, backend_type::BackendType}, data::tensor_data::TensorData, math::element::Element
};

use super::{binary::Binary, unary::Unary};

#[derive(Clone)]
pub enum Function<B: Backend> {
    U(Rc<dyn Unary<B>>),
    B(Rc<dyn Binary<B>>),
}

impl<B: Backend> Debug for Function<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U(u) => write!(f, "Unary: {}", u.tag()),
            Self::B(b) => write!(f, "Binary: {}", b.tag()),
        }
    }
}
