use std::marker::PhantomData;

use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    ops::function::Function,
    tensor::Tensor,
};

#[derive(Clone, Debug)]
pub struct History<E: Element, BT: BackendType, B: Backend<E, BT>> {
    pub last_fn: Option<Function<E, BT, B>>,
    pub ctx: Context<B>,
    pub inputs: Vec<Tensor<E, BT, B>>,
    marker: PhantomData<E>,
}

impl<E: Element, BT: BackendType, B: Backend<E, BT>> Default for History<E, BT, B> {
    fn default() -> Self {
        Self {
            last_fn: Default::default(),
            ctx: Default::default(),
            inputs: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<E: Element, BT: BackendType, B: Backend<E, BT>> History<E, BT, B> {
    pub fn last_fn(mut self, f: Function<E, BT, B>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, t: Tensor<E, BT, B>) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn context(mut self, c: Context<B>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
