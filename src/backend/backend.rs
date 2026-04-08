use std::marker::PhantomData;

use crate::{
    backend::mode::{Gpu, Par, Seq},
    math::element::Element,
    ops::ops::Ops,
    storage::{cpu_data::CpuData, data::Data, gpu_data::GpuData},
    util::{
        profiler::{NoopProfiler, Profiler},
        unsafe_usize_convert::UnsafeUsizeConvert,
    },
};

use super::mode::Mode;

// TODO: remove clone and debug constraints
pub trait Backend: Clone + std::fmt::Debug {
    type Element: Element + UnsafeUsizeConvert;
    type Mode: Mode;
    type Profiler: Profiler;
    type Storage<'a>: Ops<Self::Element, Self::Mode, Self::Profiler>
        + Data<Self::Element>
        + Clone
        + std::fmt::Debug
    where
        Self: 'a;
}

#[derive(Clone, Debug)]
pub struct CpuSeqBackend<P: Profiler = NoopProfiler> {
    _p: PhantomData<P>,
}
impl<P: Profiler + Clone + std::fmt::Debug + 'static> Backend for CpuSeqBackend<P> {
    type Element = f64;
    type Mode = Seq;
    type Profiler = P;
    type Storage<'a> = CpuData;
}

#[derive(Clone, Debug)]
pub struct CpuParBackend<P: Profiler = NoopProfiler> {
    _p: PhantomData<P>,
}
impl<P: Profiler + Clone + std::fmt::Debug + 'static> Backend for CpuParBackend<P> {
    type Element = f64;
    type Mode = Par;
    type Profiler = P;
    type Storage<'a> = CpuData;
}

#[derive(Clone, Debug)]
pub struct GpuBackend<P: Profiler = NoopProfiler> {
    _p: PhantomData<P>,
}
impl<P: Profiler + Clone + std::fmt::Debug + 'static> Backend for GpuBackend<P> {
    type Element = f32;
    type Mode = Gpu;
    type Profiler = P;
    type Storage<'a> = GpuData<'a>;
}
