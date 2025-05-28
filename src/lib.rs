#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(trait_alias)]
#![allow(dead_code)]
pub mod autodiff;
#[allow(clippy::module_inception)]
pub mod backend;
pub mod data;
pub mod math;
pub mod module;
pub mod ops;
pub mod optim;
pub mod shaping;
pub mod tensor;
pub mod training;
pub mod util;
