use crate::autodiff::context::Context;
use crate::function::unary::Unary;
use crate::tensor::tensor::Tensor;

pub struct Neg;
impl<const N: usize> Unary<Tensor<N>> for Neg {
    fn forward(&self, a: Tensor<N>) -> Tensor<N> {
        a.map(|v| -v)
    }

    fn backward(&self, _ctx: &Context<Tensor<N>, Tensor<N>>, d: Tensor<N>) -> Tensor<N> {
        d.map(|v| -v)
    }
}

pub struct Inv;
impl<const N: usize> Unary<Tensor<N>> for Inv {
    fn forward(&self, a: Tensor<N>) -> Tensor<N> {
        a.map(|v| if v == 0. { 0. } else { 1. / v })
    }

    fn backward(&self, _ctx: &Context<Tensor<N>, Tensor<N>>, _d: Tensor<N>) -> Tensor<N> {
        todo!()
    }
}
