use std::slice::Iter;

use crate::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

pub trait TensorData {
    fn shape(&self) -> &Shape;
    fn size(&self) -> usize;
    fn iter(&self) -> Iter<'_, f32>;
    fn first(&self) -> Option<f32>;
    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;
    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized;
    fn transpose(&self) -> Option<Self>
    where
        Self: Sized,
    {
        let mut order: Vec<_> = Order::range(self.shape().len())
            .data
            .iter()
            .map(|&u| u as f32)
            .collect();
        let len = order.len();
        order.swap(len - 2, len - 1);
        self.permute(&Self::vec(order))
    }
    fn index(&self, idx: Idx) -> f32;
    fn indices(&self) -> impl Iterator<Item = Idx>;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;
    fn epsilon(shape: Shape, idx: &Idx, eps: f32) -> Self;

    fn from(data: Vec<f32>, shape: Shape, strides: Strides) -> Self;
    fn scalar(s: f32) -> Self;
    fn vec(v: Vec<f32>) -> Self;
    fn matrix(m: Vec<Vec<f32>>) -> Option<Self>
    where
        Self: Sized;
}
