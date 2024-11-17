use crate::autodiff::context::Context;

use super::{scalar::Scalar, scalar_function::ScalarFunction};

#[derive(Debug, Default)]
pub struct ScalarHistory {
    pub last_fn: Option<ScalarFunction>,
    pub ctx: Context,
    pub inputs: Vec<Scalar>,
}

impl ScalarHistory {
    pub fn last_fn(mut self, f: ScalarFunction) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, s: Scalar) -> Self {
        self.inputs.push(s);
        self
    }

    pub fn context(mut self, c: Context) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
