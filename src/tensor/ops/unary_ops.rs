use crate::{
    autodiff::context::Context, function::unary::Unary, math::math_unary,
    tensor::tensor_data::TensorData,
};

pub struct Neg;
impl Unary<TensorData> for Neg {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::neg)
    }

    fn backward(&self, _ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        d.map(math_unary::neg_back)
    }
}

pub struct Inv;
impl Unary<TensorData> for Inv {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::inv)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math_unary::inv_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Ln;
impl Unary<TensorData> for Ln {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::ln)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math_unary::ln_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Sig;
impl Unary<TensorData> for Sig {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::sig)
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math_unary::sig_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Relu;
impl Unary<TensorData> for Relu {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::relu)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math_unary::relu_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Exp;
impl Unary<TensorData> for Exp {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math_unary::exp)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math_unary::exp_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}
