use crate::{data::tensor_data::TensorData, shaping::shape::Shape};

use super::backend_type::BackendType;

pub trait TensorBackend<T: BackendType> {
    fn map<F: Fn(f64) -> f64 + Sync>(&self, f: F) -> Self;
    fn map_broadcast<F: Fn(f64) -> f64 + Sync>(&self, out: &Self, f: F) -> Option<Self>
    where
        Self: Sized;
    fn zip<F: Fn(f64, f64) -> f64 + Sync>(&self, other: &Self, f: F) -> Option<Self>
    where
        Self: Sized;
    fn reduce<F: Fn(f64, f64) -> f64 + Sync>(&self, f: F, dim: usize, init: f64) -> Option<Self>
    where
        Self: Sized;
    fn matmul(&self, other: &Self) -> Self;

    fn expand(&self, other: Self) -> Option<Self>
    where
        Self: Sized + TensorData,
    {
        if self.shape() == other.shape() {
            return Some(other);
        }

        let bc_shape = self.shape().broadcast(other.shape())?;
        let buf = TensorData::zeros(bc_shape);
        let mut out = other.map_broadcast(&buf, |f| f)?;
        if self.shape() == out.shape() {
            return Some(out);
        }

        let orig_shape = Shape::new(
            [
                vec![1; out.shape().len() - self.shape().len()],
                self.shape().data().to_vec(),
            ]
            .concat(),
        );
        for (dim, shape) in out.shape().clone().data().iter().enumerate() {
            if orig_shape.data()[dim] == 1 && *shape != 1 {
                out = out.reduce(|a, b| a + b, dim, 0.)?;
            }
        }
        assert!(
            out.size() == self.size(),
            "out shape: {:?}, self shape: {:?}",
            out.shape(),
            self.shape(),
        );
        Some(out)
    }
}

pub trait Backend<T: BackendType> = TensorBackend<T> + TensorData + Clone + std::fmt::Debug;
