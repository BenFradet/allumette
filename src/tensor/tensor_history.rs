use crate::{autodiff::context::Context, function::function::Function};

use super::{tensor::Tensor, tensor_data::TensorData};

#[derive(Clone, Debug, Default)]
pub struct TensorHistory {
    pub last_fn: Option<Function<TensorData>>,
    pub ctx: Context<TensorData>,
    pub inputs: Vec<Tensor>,
}

impl TensorHistory {
    pub fn last_fn(mut self, f: Function<TensorData>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, t: Tensor) -> Self {
        self.inputs.push(t);
        self
    }

    pub fn context(mut self, c: Context<TensorData>) -> Self {
        self.ctx = c;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.last_fn.is_none() && self.ctx.is_empty()
    }
}
