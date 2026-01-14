#![feature(trait_alias)]
pub mod autodiff;
#[allow(clippy::module_inception)]
pub mod backend;
pub mod data;
pub mod fns;
pub mod math;
pub mod ops;
pub mod optim;
pub mod shaping;
pub mod tensor;
pub mod training;
pub mod util;
pub mod wgpu;
