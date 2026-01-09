use crate::{backend::backend_type::{Gpu, Par, Seq}, data::{cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData, tensor_data::TensorData}, math::element::Element, shaping::shape::Shape, util::unsafe_usize_convert::UnsafeUsizeConvert};

use super::backend_type::BackendType;

pub trait Backend: Clone + std::fmt::Debug {
    type Element: Element + UnsafeUsizeConvert;
    type BackendType: BackendType;
    type Storage<'a>: TensorBackend<Self::Element, Self::BackendType> + TensorData<Self::Element> + Clone + std::fmt::Debug
        where Self: 'a;
}

#[derive(Clone, Debug)]
pub struct CpuSeqBackend;
impl Backend for CpuSeqBackend {
    type Element = f64;
    type BackendType = Seq;
    type Storage<'a> = CpuTensorData;
}

#[derive(Clone, Debug)]
pub struct CpuParBackend;
impl Backend for CpuParBackend {
    type Element = f64;
    type BackendType = Par;
    type Storage<'a> = CpuTensorData;
}

#[derive(Clone, Debug)]
pub struct GpuBackend;
impl Backend for GpuBackend {
    type Element = f32;
    type BackendType = Gpu;
    type Storage<'a> = GpuTensorData<'a>;
}

pub trait TensorBackend<E: Element, T: BackendType> {
    fn map<F: Fn(E) -> E + Sync>(&self, f: F, tag: &'static str) -> Self;
    fn map_broadcast<F: Fn(E) -> E + Sync>(
        &self,
        out: &Self,
        f: F,
        tag: &'static str,
    ) -> Option<Self>
    where
        Self: Sized;
    fn zip<F: Fn(E, E) -> E + Sync>(&self, other: &Self, f: F, tag: &'static str) -> Option<Self>
    where
        Self: Sized;
    fn reduce<F: Fn(E, E) -> E + Sync>(
        &self,
        f: F,
        dim: usize,
        init: E,
        tag: &'static str,
    ) -> Option<Self>
    where
        Self: Sized;
    fn matmul(&self, other: &Self) -> Option<Self>
    where
        Self: Sized;

    fn expand(&self, other: Self) -> Option<Self>
    where
        Self: Sized + TensorData<E>,
    {
        if self.shape() == other.shape() {
            return Some(other);
        }

        let bc_shape = self.shape().broadcast(other.shape())?;
        let buf = TensorData::zeros(bc_shape);
        let mut out = other.map_broadcast(&buf, |f| f, "id")?;
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
                out = out.reduce(|a, b| a + b, dim, E::zero(), "sum")?;
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
