use crate::{
    autodiff::context::Context,
    function::unary::Unary,
    math,
    tensor::{shaping::shape::Shape, tensor_data::TensorData},
};

pub struct Neg;
impl Unary<TensorData> for Neg {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::neg)
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        d.map(math::unary::neg_back)
    }
}

pub struct Inv;
impl Unary<TensorData> for Inv {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::inv)
    }

    fn backward(&self, ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::inv_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Ln;
impl Unary<TensorData> for Ln {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::ln)
    }

    fn backward(&self, ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::ln_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Sig;
impl Unary<TensorData> for Sig {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::sig)
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::sig_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Relu;
impl Unary<TensorData> for Relu {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::relu)
    }

    fn backward(&self, ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::relu_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

pub struct Exp;
impl Unary<TensorData> for Exp {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map(math::unary::exp)
    }

    fn backward(&self, ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::exp_back))
            .unwrap_or(TensorData::ones(d.shape.clone()))
    }
}

// make contiguous
pub struct Copy;
impl Unary<TensorData> for Copy {
    fn forward(&self, a: &TensorData) -> TensorData {
        a.map_broadcast(&TensorData::zeros(a.shape.clone()), |f| f).unwrap_or(a.map(|f| f))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: &TensorData) -> TensorData {
        d.clone()
    }
}
