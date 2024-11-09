#[derive(Debug)]
pub struct Parameter {
    name: String,
    value: String,
}

impl Parameter {
    fn update(self, value: String) -> Self {
        Self {
            name: self.name,
            value
        }
    }
}