use super::{context::Context, scalar::Scalar, scalar_function::ScalarFunction};

#[derive(Debug)]
pub struct ScalarHistory {
    pub last_fn: Option<ScalarFunction>,
    pub ctx: Context,
    pub inputs: Vec<Scalar>,
}

impl Default for ScalarHistory {
    fn default() -> Self {
        Self {
            last_fn: None,
            ctx: Context::default(),
            inputs: vec![],
        }
    }
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
