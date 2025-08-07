use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    data::tensor_data::TensorData,
    math::{self, element::Element},
    shaping::shape::Shape,
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

use super::binary::Binary;

pub struct Add;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Add {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, math::binary::add)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (d.clone(), d.clone())
    }

    fn tag(&self) -> &str {
        "add"
    }
}

pub struct Mul;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Mul {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, math::binary::mul)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.zip(d, math::binary::mul))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
            ctx.fst
                .as_ref()
                .and_then(|a| a.zip(d, math::binary::mul))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
        )
    }

    fn tag(&self) -> &str {
        "mul"
    }
}

pub struct Lt;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Lt {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, math::binary::lt)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::zeros(d.shape().clone()),
            <T as TensorData<E>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &str {
        "lt"
    }
}

pub struct Eq;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Eq {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, math::binary::eq)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::zeros(d.shape().clone()),
            <T as TensorData<E>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &str {
        "eq"
    }
}

pub struct Sum;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Sum {
    fn forward(&self, a: &T, dim: &T) -> T {
        a.reduce(|acc, v| acc + v, dim.first().unwrap().unsafe_to(), 0.)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (d.clone(), <T as TensorData<E>>::from_scalar(E::zero()))
    }

    fn tag(&self) -> &str {
        "sum"
    }
}

pub struct Permute;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Permute {
    fn forward(&self, a: &T, order: &T) -> T {
        let a_shape = a.shape().clone();
        a.permute(order)
            .unwrap_or(<T as TensorData<E>>::ones(a_shape))
    }

    fn backward(&self, _ctx: &Context<T>, _d: &T) -> (T, T) {
        todo!();
        //let order = ctx
        //    .snd
        //    .as_ref()
        //    .map(|o| o.into())
        //    .unwrap_or(Order::range(d.dims()));
        //let mut inv = vec![];
        //for i in 0..order.len() {
        //    let idx = order.index(i).unwrap_or(0);
        //    inv.push(idx);
        //}
        //let inverse_order = Order::new(inv).unwrap_or(Order::range(order.len()));
        //(
        //    d.permute(&inverse_order)
        //        .unwrap_or(TensorData::ones(d.shape.clone())),
        //    TensorData::scalar(0.),
        //)
    }

    fn tag(&self) -> &str {
        "permute"
    }
}

pub struct IsClose;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for IsClose {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, |a, b| if math::binary::is_close(a, b) { 1. } else { 0. })
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, _d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::from_scalar(E::zero()),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &str {
        "is close"
    }
}

pub struct All;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for All {
    fn forward(&self, a: &T, dim: &T) -> T {
        a.reduce(|acc, v| acc * v, dim.first().unwrap().unsafe_to(), 1.)
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, _d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::from_scalar(E::zero()),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &str {
        "all"
    }
}

pub struct View;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T>
    for View
{
    fn forward(&self, lhs: &T, s: &T) -> T {
        assert!(lhs.is_contiguous(), "must be contiguous to view");
        let shape = Shape::new(s.collect().iter().map(|f| f.unsafe_to()).collect());
        lhs.reshape(shape)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        let shape = ctx
            .fst
            .as_ref()
            .map(|o| o.shape().clone())
            .unwrap_or(d.shape().clone());
        (
            d.reshape(shape),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &str {
        "view"
    }
}

pub struct MatMul;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for MatMul {
    fn forward(&self, lhs: &T, rhs: &T) -> T {
        lhs.matmul(rhs)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.transpose())
                .map(|b| d.matmul(&b))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
            ctx.fst
                .as_ref()
                .and_then(|a| a.transpose())
                .map(|a| a.matmul(d))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
        )
    }

    fn tag(&self) -> &str {
        "matmul"
    }
}
