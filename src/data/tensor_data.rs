use crate::{
    math::element::Element,
    shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides},
};

pub trait TensorData<E: Element> {
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &Strides;
    fn size(&self) -> usize;

    // expensive, use with parsimony (cpu: allocate, gpu: retrieve to cpu)
    fn collect(&self) -> Vec<E>;
    fn first(&self) -> Option<E>;

    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;
    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized;
    fn transpose(&self) -> Option<Self>
    where
        Self: Sized;

    fn indices(&self) -> impl Iterator<Item = Idx>;

    fn to_order(&self) -> Order;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;
    fn rand_with_seed(shape: Shape, seed: u64) -> Self;
    fn epsilon(shape: Shape, idx: &Idx, eps: E) -> Self;

    fn from(data: &[E], shape: Shape, strides: Strides) -> Self;
    fn from_scalar(s: E) -> Self;
    fn from_1d(v: &[E]) -> Self;
    fn from_2d(m: &[&[E]]) -> Option<Self>
    where
        Self: Sized;
}
