use crate::autodiff::context::Context;
use crate::function::unary::Unary;
use crate::tensor::tensor_data_n::TensorDataN;

pub struct Neg;
impl<const N: usize> Unary<TensorDataN<N>> for Neg {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| -v)
    }

    fn backward(
        &self,
        _ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        d.map(|v| -v)
    }
}

pub struct Inv;
impl<const N: usize> Unary<TensorDataN<N>> for Inv {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| if v == 0. { 0. } else { 1. / v })
    }

    fn backward(
        &self,
        ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        let a = ctx.a.clone().unwrap_or(TensorDataN::ones(d.shape.clone()));
        d.zip_n(&a, |d, a| {
            let ap = if a == 0. { 1. } else { a };
            -d / ap.powf(2.)
        })
    }
}

pub struct Ln;
impl<const N: usize> Unary<TensorDataN<N>> for Ln {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| if v <= 0. { 0. } else { v.ln() })
    }

    fn backward(
        &self,
        ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        let a = ctx.a.clone().unwrap_or(TensorDataN::ones(d.shape.clone()));
        d.zip_n(&a, |d, a| {
            let ap = if a == 0. { 1. } else { a };
            d / ap
        })
    }
}

pub struct Sig;
impl<const N: usize> Unary<TensorDataN<N>> for Sig {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| {
            if v >= 0. {
                1. / (1. + (-v).exp())
            } else {
                v.exp() / (1. + v.exp())
            }
        })
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(
        &self,
        ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        let a = ctx.a.clone().unwrap_or(TensorDataN::zeros(d.shape.clone()));
        let sig_a = self.forward(a);
        d.zip_n(&sig_a, |d, a| a * (1. - a) * d)
    }
}

pub struct Relu;
impl<const N: usize> Unary<TensorDataN<N>> for Relu {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| v.max(0.))
    }

    fn backward(
        &self,
        ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        let a = ctx.a.clone().unwrap_or(TensorDataN::zeros(d.shape.clone()));
        d.zip_n(&a, |d, a| if a > 0. { d } else { 0. })
    }
}

pub struct Exp;
impl<const N: usize> Unary<TensorDataN<N>> for Exp {
    fn forward(&self, a: TensorDataN<N>) -> TensorDataN<N> {
        a.map(|v| v.exp())
    }

    fn backward(
        &self,
        ctx: &Context<TensorDataN<N>, TensorDataN<N>>,
        d: TensorDataN<N>,
    ) -> TensorDataN<N> {
        let a = ctx.a.clone().unwrap_or(TensorDataN::zeros(d.shape.clone()));
        d.zip_n(&a, |d, a| a.exp() * d)
    }
}
