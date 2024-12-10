use crate::scalar::scalar::Scalar;

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub scalar: Scalar,
}

impl Parameter {
    pub fn new(scalar: Scalar) -> Self {
        let id = scalar.id.clone();
        Self { name: id, scalar }
    }
}
