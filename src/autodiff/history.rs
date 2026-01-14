use std::marker::PhantomData;

use crate::{
    autodiff::context::Context,
    backend::backend::Backend,
    fns::function::Function,
    tensor::Tensor,
};

#[derive(Clone, Debug)]
pub struct History<'a, B: Backend> {
    pub last_fn: Option<Function<'a, B>>,
    pub ctx: Context<B::Storage<'a>>,
    pub inputs: Vec<Tensor<'a, B>>,
    _marker: PhantomData<B::Element>,
}

impl<'a, B: Backend> Default for History<'a, B> {
    fn default() -> Self {
        Self {
            last_fn: Default::default(),
            ctx: Default::default(),
            inputs: Default::default(),
            _marker: PhantomData,
        }
    }
}

impl<'a, B: Backend> History<'a, B> {
    pub fn last_fn(mut self, f: Function<'a, B>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, t: Tensor<'a, B>) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn context(mut self, c: Context<B::Storage<'a>>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
