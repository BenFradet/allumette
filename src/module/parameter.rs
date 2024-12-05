use crate::variable::Variable;

pub struct Parameter {
    pub name: String,
    pub var: Box<dyn Variable>,
}

impl Parameter {
}
