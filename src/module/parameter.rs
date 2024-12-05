use crate::scalar::scalar::Scalar;

#[derive(Debug)]
pub struct Parameter {
    pub name: String,
    pub scalar: Scalar,
}

impl Parameter {
    fn var(mut self, scalar: Scalar) -> Self {
        self.scalar = scalar;
        self
    }
}
