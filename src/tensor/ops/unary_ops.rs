use crate::{autodiff::context::Context, function::unary::Unary, tensor::tensor_data::TensorData, util::math};

pub struct Neg;
impl Unary<TensorData> for Neg {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::neg)
    }

    fn backward(&self, _ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        d.map(math::neg_back)
    }
}

pub struct Inv;
impl Unary<TensorData> for Inv {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::inv)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math::inv_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Ln;
impl Unary<TensorData> for Ln {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::ln)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math::ln_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Sig;
impl Unary<TensorData> for Sig {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::sig)
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math::sig_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Relu;
impl Unary<TensorData> for Relu {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::relu)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math::relu_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Exp;
impl Unary<TensorData> for Exp {
    fn forward(&self, a: TensorData) -> TensorData {
        a.map(math::exp)
    }

    fn backward(&self, ctx: &Context<TensorData, TensorData>, d: TensorData) -> TensorData {
        ctx.a
            .as_ref()
            .and_then(|a| a.zip(&d, math::exp_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}
