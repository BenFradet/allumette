use crate::{autodiff::context::Context, function::function::Function};

use super::scalar::Scalar;

#[derive(Clone, Debug, Default)]
pub struct ScalarHistory {
    pub last_fn: Option<Function<f64>>,
    pub ctx: Context<f64>,
    pub inputs: Vec<Scalar>,
}

impl ScalarHistory {
    pub fn last_fn(mut self, f: Function<f64>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, s: Scalar) -> Self {
        self.inputs.push(s);
        self
    }

    pub fn context(mut self, c: Context<f64>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
