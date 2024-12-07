use crate::scalar::scalar::Scalar;

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub scalar: Scalar,
}

impl Parameter {
    pub fn new(name: String, scalar: Scalar) -> Self {
        Self {
            name,
            scalar,
        }
    }

    pub fn scalar(mut self, scalar: Scalar) -> Self {
        self.scalar = scalar;
        self
    }
}
