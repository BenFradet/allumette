use crate::tensor::{shaping::shape::Shape, tensor_data::TensorData};
use std::slice::Iter;

pub trait Shaped {
    fn shape(&self) -> Shape;
    fn size(&self) -> usize;
    fn iter(&self) -> Iter<'_, f64>;
    fn first(&self) -> Option<f64>;
    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;

    fn scalar(s: f64) -> Self;
    fn vec(v: Vec<f64>) -> Self;
    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized;
}

impl Shaped for TensorData {
    fn shape(&self) -> Shape {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }

    fn iter(&self) -> Iter<'_, f64> {
        todo!()
    }

    fn first(&self) -> Option<f64> {
        todo!()
    }

    fn is_contiguous(&self) -> bool {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Self {
        todo!()
    }

    fn ones(shape: Shape) -> Self {
        todo!()
    }

    fn zeros(shape: Shape) -> Self {
        todo!()
    }

    fn rand(shape: Shape) -> Self {
        todo!()
    }

    fn scalar(s: f64) -> Self {
        todo!()
    }

    fn vec(v: Vec<f64>) -> Self {
        todo!()
    }

    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
