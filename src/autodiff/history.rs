use crate::{autodiff::context::Context, function::function::Function};

#[derive(Clone, Debug, Default)]
pub struct History<A> {
    pub last_fn: Option<Function<A>>,
    pub ctx: Context<A>,
    pub inputs: Vec<A>,
}

impl<A: Clone> History<A> {
    pub fn last_fn(mut self, f: Function<A>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, a: A) -> Self {
        self.inputs.push(a);
        self
    }

    pub fn context(mut self, c: Context<A>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
