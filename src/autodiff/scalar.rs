use super::scalar_history::ScalarHistory;

// TODO: abstract over f64
pub struct Scalar {
    history: Option<ScalarHistory>,
}