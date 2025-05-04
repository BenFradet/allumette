use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    function::function::Function,
    tensor::Tensor,
};

#[derive(Clone, Debug)]
pub struct History<BT: BackendType, B: Backend<BT>> {
    pub last_fn: Option<Function<BT, B>>,
    pub ctx: Context<B>,
    pub inputs: Vec<Tensor<BT, B>>,
}

impl<BT: BackendType, B: Backend<BT>> Default for History<BT, B> {
    fn default() -> Self {
        Self {
            last_fn: Default::default(),
            ctx: Default::default(),
            inputs: Default::default(),
        }
    }
}

impl<BT: BackendType, B: Backend<BT> + Clone> History<BT, B> {
    pub fn last_fn(mut self, f: Function<BT, B>) -> Self {
        self.last_fn = Some(f);
        self
    }

    pub fn push_input(mut self, t: Tensor<BT, B>) -> Self {
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
