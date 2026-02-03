use crate::{
    backend::mode::{Gpu, Par, Seq},
    math::element::Element,
    ops::ops::Ops,
    storage::{cpu_data::CpuData, data::Data, gpu_data::GpuData},
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

use super::mode::Mode;

// TODO: remove clone and debug constraints
pub trait Backend: Clone + std::fmt::Debug {
    type Element: Element + UnsafeUsizeConvert;
    type Mode: Mode;
    type Storage<'a>: Ops<Self::Element, Self::Mode> + Data<Self::Element> + Clone + std::fmt::Debug
    where
        Self: 'a;
}

#[derive(Clone, Debug)]
pub struct CpuSeqBackend;
impl Backend for CpuSeqBackend {
    type Element = f64;
    type Mode = Seq;
    type Storage<'a> = CpuData;
}

#[derive(Clone, Debug)]
pub struct CpuParBackend;
impl Backend for CpuParBackend {
    type Element = f64;
    type Mode = Par;
    type Storage<'a> = CpuData;
}

#[derive(Clone, Debug)]
pub struct GpuBackend;
impl Backend for GpuBackend {
    type Element = f32;
    type Mode = Gpu;
    type Storage<'a> = GpuData<'a>;
}
