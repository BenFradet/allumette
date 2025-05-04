use std::slice::Iter;

use crate::shaping::shape::Shape;

pub trait TensorData {
    fn shape(&self) -> &Shape;
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
