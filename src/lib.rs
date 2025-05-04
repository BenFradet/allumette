#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![allow(dead_code)]
pub mod autodiff;
#[allow(clippy::module_inception)]
pub mod backend;
pub mod data;
#[allow(clippy::module_inception)]
pub mod function;
pub mod math;
pub mod module;
pub mod ops;
pub mod optim;
pub mod shaping;
pub mod tensor;
pub mod train;
pub mod util;
