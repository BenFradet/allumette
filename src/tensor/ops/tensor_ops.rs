pub trait TensorOps {
    type Placeholder;

    fn map(t: &Self::Placeholder, f: impl Fn(f64) -> f64) -> Self::Placeholder;
    fn zip(t: &Self::Placeholder, f: impl Fn(f64, f64) -> f64) -> Self::Placeholder;
    fn reduce(t: &Self::Placeholder, f: impl Fn(f64, f64) -> f64) -> Self::Placeholder;
}