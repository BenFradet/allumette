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

    // TODO: context update
}
