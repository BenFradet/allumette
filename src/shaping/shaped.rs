use crate::tensor::shaping::shape::Shape;

pub trait Shaped {
    fn shape(&self) -> Shape;
    fn first(&self) -> Option<f64>;
    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;
    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn scalar(s: f64) -> Self;
}
