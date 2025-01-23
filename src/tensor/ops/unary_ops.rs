use crate::function::unary::Unary;
use crate::autodiff::context::Context;
use crate::tensor::tensor::Tensor;

pub struct Neg;
impl<const N: usize> Unary<Tensor<N>> for Neg {
    fn forward(&self, a: Tensor<N>) -> Tensor<N> {
        a.map(|v| -v)
    }

    fn backward(&self, _ctx: &Context<Tensor<N>>, d: Tensor<N>) -> Tensor<N> {
        d.map(|v| -v)
    }
}

pub struct Inv;
impl<const N: usize> Unary<Tensor<N>> for Inv {
    fn forward(&self, a: Tensor<N>) -> Tensor<N> {
        a.map(|v| {
            if v == 0. {
                0.
            } else {
                1. / v
            }
        })
    }

    fn backward(&self, ctx: &Context<Tensor<N>>, d: Tensor<N>) -> Tensor<N> {
        let vs = &ctx.saved_values;
        let a = vs.first().filter(|v| **v != 0.).unwrap_or(&1.);
        (-1. / (a.powf(2.))) * d
    }
}
