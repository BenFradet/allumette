use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    data::tensor_data::TensorData,
    math::element::Element,
    shaping::{order::Order, shape::Shape},
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

use super::binary::Binary;

pub struct Add;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Add {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, |e1, e2| e1 + e2, <Add as Binary<E, BT, T>>::tag(self))
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (d.clone(), d.clone())
    }

    fn tag(&self) -> &'static str {
        "add"
    }
}

pub struct Mul;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Mul {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(b, |e1, e2| e1 * e2, <Mul as Binary<E, BT, T>>::tag(self))
            .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.zip(d, |e1, e2| e1 * e2, <Mul as Binary<E, BT, T>>::tag(self)))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
            ctx.fst
                .as_ref()
                .and_then(|a| a.zip(d, |e1, e2| e1 * e2, <Mul as Binary<E, BT, T>>::tag(self)))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
        )
    }

    fn tag(&self) -> &'static str {
        "mul"
    }
}

pub struct Lt;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Lt {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(
            b,
            |e1, e2| if e1 < e2 { E::one() } else { E::zero() },
            <Lt as Binary<E, BT, T>>::tag(self),
        )
        .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::zeros(d.shape().clone()),
            <T as TensorData<E>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &'static str {
        "lt"
    }
}

pub struct Eq;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Eq {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(
            b,
            |e1, e2| if e1 == e2 { E::one() } else { E::zero() },
            <Eq as Binary<E, BT, T>>::tag(self),
        )
        .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::zeros(d.shape().clone()),
            <T as TensorData<E>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &'static str {
        "eq"
    }
}

pub struct IsClose;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for IsClose {
    fn forward(&self, a: &T, b: &T) -> T {
        a.zip(
            b,
            |e1, e2| if e1.is_close(e2) { E::one() } else { E::zero() },
            <IsClose as Binary<E, BT, T>>::tag(self),
        )
        .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, _d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::from_scalar(E::zero()),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "is_close"
    }
}

pub struct Sum;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for Sum {
    fn forward(&self, a: &T, dim: &T) -> T {
        a.reduce(
            |acc, v| acc + v,
            dim.first().unwrap_or(E::one()).unsafe_to(),
            E::zero(),
            <Sum as Binary<E, BT, T>>::tag(self),
        )
        .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> (T, T) {
        (d.clone(), <T as TensorData<E>>::from_scalar(E::zero()))
    }

    fn tag(&self) -> &'static str {
        "sum"
    }
}

pub struct All;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for All {
    fn forward(&self, a: &T, dim: &T) -> T {
        a.reduce(
            |acc, v| acc * v,
            dim.first().unwrap().unsafe_to(),
            E::one(),
            <All as Binary<E, BT, T>>::tag(self),
        )
        .unwrap_or(<T as TensorData<E>>::ones(a.shape().clone()))
    }

    fn backward(&self, _ctx: &Context<T>, _d: &T) -> (T, T) {
        (
            <T as TensorData<E>>::from_scalar(E::zero()),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "all"
    }
}

pub struct Permute;
impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T>
    for Permute
{
    fn forward(&self, a: &T, order: &T) -> T {
        let a_shape = a.shape().clone();
        a.permute(order)
            .unwrap_or(<T as TensorData<E>>::ones(a_shape))
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        let order = ctx
            .snd
            .as_ref()
            .map(|o| o.to_order())
            .unwrap_or(Order::range(d.shape().len()));
        let mut inv = vec![];
        for i in 0..order.len() {
            let idx = order.index(i).unwrap_or(0);
            inv.push(idx);
        }
        let inverse_order = Order::new(inv).unwrap_or(Order::range(order.len()));
        let inverse_data: Vec<_> = inverse_order
            .data
            .iter()
            .map(|u| E::unsafe_from(*u))
            .collect();
        let inverse_order_td = <T as TensorData<E>>::from_1d(&inverse_data);
        (
            d.permute(&inverse_order_td)
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
            <T as TensorData<E>>::from_scalar(E::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "permute"
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

    // TODO: id tag for gpu, improve
    fn tag(&self) -> &'static str {
        "id"
    }
}

pub struct MatMul;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Binary<E, BT, T> for MatMul {
    fn forward(&self, lhs: &T, rhs: &T) -> T {
        lhs.matmul(rhs)
            .unwrap_or(<T as TensorData<E>>::zeros(lhs.shape().clone()))
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> (T, T) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.transpose())
                .and_then(|b| d.matmul(&b))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
            ctx.fst
                .as_ref()
                .and_then(|a| a.transpose())
                .and_then(|a| a.matmul(d))
                .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone())),
        )
    }

    fn tag(&self) -> &'static str {
        "mm"
    }
}
