use crate::{
    backend::mode::{Gpu, Par, Seq}, data::{
        cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData, tensor_data::TensorData,
    }, math::element::Element, ops::tensor_ops::Ops, util::unsafe_usize_convert::UnsafeUsizeConvert
};

use super::mode::Mode;

// TODO: remove clone and debug constraints
pub trait Backend: Clone + std::fmt::Debug {
    type Element: Element + UnsafeUsizeConvert;
    type Mode: Mode;
    type Storage<'a>: Ops<Self::Element, Self::Mode>
        + TensorData<Self::Element>
        + Clone
        + std::fmt::Debug
    where
        Self: 'a;
}

#[derive(Clone, Debug)]
pub struct CpuSeqBackend;
impl Backend for CpuSeqBackend {
    type Element = f64;
    type Mode = Seq;
    type Storage<'a> = CpuTensorData;
}

#[derive(Clone, Debug)]
pub struct CpuParBackend;
impl Backend for CpuParBackend {
    type Element = f64;
    type Mode = Par;
    type Storage<'a> = CpuTensorData;
}

#[derive(Clone, Debug)]
pub struct GpuBackend;
impl Backend for GpuBackend {
    type Element = f32;
    type Mode = Gpu;
    type Storage<'a> = GpuTensorData<'a>;
}
