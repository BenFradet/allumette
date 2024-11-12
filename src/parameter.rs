use proptest::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub value: f64,
}

impl Parameter {
    fn update(self, value: f64) -> Self {
        Self {
            name: self.name,
            value
        }
    }

    pub fn arb() -> impl Strategy<Value = Parameter> {
        (
            ".*",
            any::<f64>(),
        ).prop_map(|(name, value)| Parameter { name, value} )
    }
}