use crate::{autodiff::context::Context, function::unary::Unary, tensor::tensor_data::TensorData};

pub struct Neg;
impl Unary<TensorData> for Neg {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(|v| -v)
    }

    fn backward(
        &self,
        _ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        d.map(|v| -v)
    }
}

pub struct Inv;
impl Unary<TensorData> for Inv {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(|v| if v == 0. { 0. } else { 1. / v })
    }

    fn backward(
        &self,
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        ctx.a.as_ref().and_then(|a| d.zip(a, |d, a| {
            let ap = if a == 0. { 1. } else { a };
            -d / ap.powf(2.)
        })).unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Ln;
impl Unary<TensorData> for Ln {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(|v| if v <= 0. { 0. } else { v.ln() })
    }

    fn backward(
        &self,
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        ctx.a.as_ref().and_then(|a| d.zip(a, |d, a| {
            let ap = if a == 0. { 1. } else { a };
            d / ap
        })).unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Sig;
impl Unary<TensorData> for Sig {
    fn forward(&self, a: TensorData) -> TensorData {
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
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        ctx.a.as_ref().and_then(|a| d.zip(a, |d, a| a * (1. - a) * d)).unwrap_or(TensorData::zeros(d.shape.clone()))
    }
}

pub struct Relu;
impl Unary<TensorData> for Relu {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(|v| v.max(0.))
    }

    fn backward(
        &self,
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        ctx.a.as_ref().and_then(|a| d.zip(a, |d, a| if a > 0. { d } else { 0. })).unwrap_or(TensorData::zeros(d.shape.clone()))
    }
}

pub struct Exp;
impl Unary<TensorData> for Exp {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(|v| v.exp())
    }

    fn backward(
        &self,
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> TensorData {
        ctx.a.as_ref().and_then(|a| d.zip(a, |d, a| a.exp() * d)).unwrap_or(TensorData::zeros(d.shape.clone()))
    }
}
