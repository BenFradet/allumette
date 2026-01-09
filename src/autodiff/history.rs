use std::marker::PhantomData;

use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    ops::function::Function,
    tensor::Tensor,
};

#[derive(Clone, Debug)]
pub struct History<B: Backend> {
    pub last_fn: Option<Function<B>>,
    pub ctx: Context<B::Storage>,
    pub inputs: Vec<Tensor<B>>,
    _marker: PhantomData<B::Element>,
}

impl<B: Backend> Default for History<B> {
    fn default() -> Self {
        Self {
            last_fn: Default::default(),
            ctx: Default::default(),
            inputs: Default::default(),
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> History<B> {
    pub fn last_fn(mut self, f: Function<B>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, t: Tensor<B>) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn context(mut self, c: Context<B::Storage>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
