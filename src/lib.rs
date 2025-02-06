#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![allow(dead_code)]
pub mod autodiff;
pub mod data;
#[allow(clippy::module_inception)]
pub mod function;
pub mod hof;
pub mod math;
pub mod module;
pub mod optim;
pub mod scalar;
pub mod tensor;
pub mod util;
