use crate::{
    autodiff::context::Context,
    function::binary::Binary,
    math,
    tensor::{
        shaping::{order::Order, shape::Shape},
        tensor_data::TensorData,
    },
};

pub struct Add;
impl Binary<TensorData> for Add {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::add)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (d.clone(), d)
    }
}

pub struct Mul;
impl Binary<TensorData> for Mul {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::mul)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.zip(&d, math::binary::mul))
                .unwrap_or(TensorData::ones(d.shape.clone())),
            ctx.fst
                .as_ref()
                .and_then(|a| a.zip(&d, math::binary::mul))
                .unwrap_or(TensorData::ones(d.shape.clone())),
        )
    }
}

pub struct Lt;
impl Binary<TensorData> for Lt {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::lt)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape.clone()),
        )
    }
}

pub struct Eq;
impl Binary<TensorData> for Eq {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(&b, math::binary::eq)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape.clone()),
        )
    }
}

pub struct Sum;
impl Binary<TensorData> for Sum {
    fn forward(&self, a: TensorData, dim: TensorData) -> TensorData {
        a.reduce(|acc, v| acc + v, dim.data[0] as usize, 0.)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        let d_shape = d.shape.clone();
        (d, TensorData::zeros(d_shape))
    }
}

pub struct Permute;
impl Binary<TensorData> for Permute {
    fn forward(&self, a: TensorData, order: TensorData) -> TensorData {
        let ord: Order = order.into();
        let a_shape = a.shape.clone();
        a.permute(&ord).unwrap_or(TensorData::ones(a_shape))
    }

    fn backward(&self, ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        let order = ctx
            .snd
            .as_ref()
            .map(|o| o.into())
            .unwrap_or(Order::range(d.dims()));
        let mut inv = vec![];
        for i in 0..order.len() {
            let idx = order.index(i).unwrap_or(0);
            inv.push(idx);
        }
        let inverse_order = Order::new(inv).unwrap_or(Order::range(order.len()));
        let d_shape = d.shape.clone();
        (
            d.permute(&inverse_order)
                .unwrap_or(TensorData::ones(d_shape.clone())),
            TensorData::zeros(d_shape),
        )
    }
}

pub struct IsClose;
impl Binary<TensorData> for IsClose {
    fn forward(&self, a: TensorData, b: TensorData) -> TensorData {
        a.zip(
            &b,
            |a, b| if math::binary::is_close(a, b) { 1. } else { 0. },
        )
        .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape),
        )
    }
}

pub struct All;
impl Binary<TensorData> for All {
    fn forward(&self, a: TensorData, dim: TensorData) -> TensorData {
        a.reduce(|acc, v| acc * v, dim.data[0] as usize, 1.)
            .unwrap_or(TensorData::ones(a.shape.clone()))
    }

    fn backward(&self, _ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        (
            TensorData::zeros(d.shape.clone()),
            TensorData::zeros(d.shape.clone()),
        )
    }
}

pub struct View;
impl Binary<TensorData> for View {
    fn forward(&self, lhs: TensorData, s: TensorData) -> TensorData {
        assert!(lhs.is_contiguous(), "must be contiguous to view");
        let shape = Shape::new(s.data.iter().map(|f| *f as usize).collect());
        lhs.shape(shape)
    }

    fn backward(&self, ctx: &Context<TensorData>, d: TensorData) -> (TensorData, TensorData) {
        let d_shape = d.shape.clone();
        let zeros = TensorData::zeros(d.shape.clone());
        let shape = ctx.fst.as_ref().map(|o| o.shape.clone()).unwrap_or(d_shape);
        (d.shape(shape), zeros)
    }
}
