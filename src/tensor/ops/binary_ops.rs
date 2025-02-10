use crate::{
    autodiff::context::Context, function::binary::Binary, math, tensor::tensor_data::TensorData,
};

pub struct Add;
impl Binary<TensorData, TensorData> for Add {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::add)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(
        &self,
        _ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> (TensorData, TensorData) {
        (d.clone(), d)
    }
}

pub struct Mul;
impl Binary<TensorData, TensorData> for Mul {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::mul)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(
        &self,
        ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> (TensorData, TensorData) {
        (
            ctx.b
                .as_ref()
                .and_then(|b| b.zip(&d, math::binary::mul))
                .unwrap_or(TensorData::ones(d.shape.clone())),
            ctx.a
                .as_ref()
                .and_then(|a| a.zip(&d, math::binary::mul))
                .unwrap_or(TensorData::ones(d.shape.clone())),
        )
    }
}

pub struct Lt;
impl Binary<TensorData, TensorData> for Lt {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::lt)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(
        &self,
        _ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape.clone()),
        )
    }
}

pub struct Eq;
impl Binary<TensorData, TensorData> for Eq {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::eq)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(
        &self,
        _ctx: &Context<TensorData, TensorData>,
        d: TensorData,
    ) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape.clone()),
        )
    }
}

pub struct Sum;
impl Binary<TensorData, TensorData> for Sum {
    fn forward(&self, a: TensorData, dim: TensorData) -> TensorData {
        a.reduce(|acc, v| acc + v, dim.data[0] as usize)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData, TensorData>, d: TensorData) -> (TensorData, TensorData) {
        let d_shape = d.shape.clone();
        (d, TensorData::zeros(d_shape))
    }
}
