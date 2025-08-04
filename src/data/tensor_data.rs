use crate::{
    math::element::Element,
    shaping::{idx::Idx, shape::Shape, strides::Strides},
};

pub trait TensorData<E: Element> {
    fn shape(&self) -> &Shape;
    fn size(&self) -> usize;

    // expensive, use with parsimony (cpu: allocate, gpu: retrieve to cpu)
    fn collect(&self) -> Vec<E>;
    // TODO: remove these as much as possible since they require data to be on the CPU
    fn first(&self) -> Option<E>;
    fn index(&self, idx: Idx) -> E;

    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;
    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized;
    fn transpose(&self) -> Option<Self>
    where
        Self: Sized;

    fn indices(&self) -> impl Iterator<Item = Idx>;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;
    fn epsilon(shape: Shape, idx: &Idx, eps: E) -> Self;

    fn from(data: Vec<E>, shape: Shape, strides: Strides) -> Self;
    fn scalar(s: E) -> Self;
    fn vec(v: Vec<E>) -> Self;
    fn matrix(m: Vec<Vec<E>>) -> Option<Self>
    where
        Self: Sized;
}
