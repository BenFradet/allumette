use super::{context::Context, scalar::Scalar, scalar_function::ScalarFunction};

pub struct ScalarHistory<'a> {
    last_fn: Option<ScalarFunction>,
    ctx: Option<Context<'a>>,
    inputs: Vec<Scalar<'a>>,
}

impl<'a> Default for ScalarHistory<'a> {
    fn default() -> Self {
        Self {
            last_fn: None,
            ctx: None,
            inputs: vec![],
        }
    }
}