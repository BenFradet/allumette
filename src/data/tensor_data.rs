use std::slice::Iter;

use crate::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

pub trait TensorData {
    fn shape(&self) -> &Shape;
    fn size(&self) -> usize;
    fn iter(&self) -> Iter<'_, f64>;
    fn first(&self) -> Option<f64>;
    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;
    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized;
    fn transpose(&self) -> Option<Self>
    where
        Self: Sized {
        let mut order: Vec<_> = Order::range(self.shape().len()).data.iter().map(|&u| u as f64).collect();
        let len = order.len();
        order.swap(len - 2, len - 1);
        self.permute(&Self::vec(order))
    }
    fn index(&self, idx: Idx) -> f64;
    fn indices(&self) -> impl Iterator<Item = Idx>;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;
    fn epsilon(shape: Shape, idx: &Idx, eps: f64) -> Self;

    fn from(data: Vec<f64>, shape: Shape, strides: Strides) -> Self;
    fn scalar(s: f64) -> Self;
    fn vec(v: Vec<f64>) -> Self;
    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized;
}
