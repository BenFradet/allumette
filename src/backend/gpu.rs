use crate::{
    backend::{backend::TensorBackend, backend_type::Gpu},
    data::gpu_tensor_data::GpuTensorData,
};

impl TensorBackend<f32, Gpu> for GpuTensorData<'_> {
    fn map<F: Fn(f32) -> f32 + Sync>(&self, f: F, tag: &str) -> Self {
        todo!()
    }

    fn map_broadcast<F: Fn(f32) -> f32 + Sync>(&self, out: &Self, f: F, tag: &str) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn zip<F: Fn(f32, f32) -> f32 + Sync>(&self, other: &Self, f: F, tag: &str) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn reduce<F: Fn(f32, f32) -> f32 + Sync>(&self, f: F, dim: usize, init: f64) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn matmul(&self, other: &Self) -> Self {
        todo!()
    }
}
