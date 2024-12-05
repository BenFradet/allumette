use proptest::prelude::*;

// TODO: remove
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub value: f64,
}

impl Parameter {
    fn update(mut self, value: f64) -> Self {
        self.value = value;
        self
    }

    pub fn arb() -> impl Strategy<Value = Parameter> {
        (".*", any::<f64>()).prop_map(|(name, value)| Parameter { name, value })
    }
}
