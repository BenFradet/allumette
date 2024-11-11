use proptest::prelude::*;

#[derive(Debug, Clone)]
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

    pub fn arb() -> impl Strategy<Value = Parameter> {
        (
            ".*",
            ".*",
        ).prop_map(|(name, value)| Parameter { name, value} )
    }
}