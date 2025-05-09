use super::backend_type::BackendType;

pub trait Backend<T: BackendType> {
    fn map<F: Fn(f64) -> f64 + Send + Sync>(&self, f: F) -> Self;
    fn map_broadcast(&self, out: &Self, f: impl Fn(f64) -> f64) -> Option<Self>
    where
        Self: Sized;
    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Option<Self>
    where
        Self: Sized;
    fn reduce(&self, f: impl Fn(f64, f64) -> f64, dim: usize, init: f64) -> Option<Self>
    where
        Self: Sized;
}
