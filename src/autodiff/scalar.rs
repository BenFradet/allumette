use super::scalar_history::ScalarHistory;

// TODO: abstract over f64
pub struct Scalar<'a> {
    v: f64,
    derivative: Option<f64>,
    history: ScalarHistory<'a>,
    id: u64,
    name: String,
}

impl<'a> Scalar<'a> {
    pub fn new(v: f64, name: String) -> Self {
        Self {
            v,
            derivative: None,
            history: ScalarHistory::default(),
            id: 0,
            name,
        }
    }
}