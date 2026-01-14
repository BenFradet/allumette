use crate::{autodiff::context::Context, backend::backend::Backend};

use crate::{
    ops::tensor_ops::Ops,
    backend::mode::Mode,
    data::tensor_data::TensorData,
    math::element::Element,
    shaping::{order::Order, shape::Shape},
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

pub trait Binary<'a, B: Backend> {
    fn forward(&self, lhs: &B::Storage<'a>, rhs: &B::Storage<'a>) -> B::Storage<'a>;
    fn backward(
        &self,
        ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>);
    fn tag(&self) -> &'static str;
}

pub struct Add;
impl<'a, B: Backend> Binary<'a, B> for Add {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(b, |e1, e2| e1 + e2, <Add as Binary<'a, B>>::tag(self))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                a.shape().clone(),
            ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (d.clone(), d.clone())
    }

    fn tag(&self) -> &'static str {
        "add"
    }
}

pub struct Mul;
impl<'a, B: Backend> Binary<'a, B> for Mul {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(b, |e1, e2| e1 * e2, <Mul as Binary<'a, B>>::tag(self))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                a.shape().clone(),
            ))
    }

    fn backward(
        &self,
        ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.zip(d, |e1, e2| e1 * e2, <Mul as Binary<'a, B>>::tag(self)))
                .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                    d.shape().clone(),
                )),
            ctx.fst
                .as_ref()
                .and_then(|a| a.zip(d, |e1, e2| e1 * e2, <Mul as Binary<'a, B>>::tag(self)))
                .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                    d.shape().clone(),
                )),
        )
    }

    fn tag(&self) -> &'static str {
        "mul"
    }
}

pub struct Lt;
impl<'a, B: Backend> Binary<'a, B> for Lt {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(
            b,
            |e1, e2| {
                if e1 < e2 {
                    B::Element::one()
                } else {
                    B::Element::zero()
                }
            },
            <Lt as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as TensorData<B::Element>>::zeros(d.shape().clone()),
            <B::Storage<'a> as TensorData<B::Element>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &'static str {
        "lt"
    }
}

pub struct Eq;
impl<'a, B: Backend> Binary<'a, B> for Eq {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(
            b,
            |e1, e2| {
                if e1 == e2 {
                    B::Element::one()
                } else {
                    B::Element::zero()
                }
            },
            <Eq as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as TensorData<B::Element>>::zeros(d.shape().clone()),
            <B::Storage<'a> as TensorData<B::Element>>::zeros(d.shape().clone()),
        )
    }

    fn tag(&self) -> &'static str {
        "eq"
    }
}

pub struct IsClose;
impl<'a, B: Backend> Binary<'a, B> for IsClose {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(
            b,
            |e1, e2| {
                if e1.is_close(e2) {
                    B::Element::one()
                } else {
                    B::Element::zero()
                }
            },
            <IsClose as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        _d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "is_close"
    }
}

pub struct Sum;
impl<'a, B: Backend> Binary<'a, B> for Sum {
    fn forward(&self, a: &B::Storage<'a>, dim: &B::Storage<'a>) -> B::Storage<'a> {
        a.reduce(
            |acc, v| acc + v,
            dim.first().unwrap_or(B::Element::one()).unsafe_to(),
            B::Element::zero(),
            <Sum as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            d.clone(),
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "sum"
    }
}

pub struct All;
impl<'a, B: Backend> Binary<'a, B> for All {
    fn forward(&self, a: &B::Storage<'a>, dim: &B::Storage<'a>) -> B::Storage<'a> {
        a.reduce(
            |acc, v| acc * v,
            dim.first().unwrap().unsafe_to(),
            B::Element::one(),
            <All as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _ctx: &Context<B::Storage<'a>>,
        _d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "all"
    }
}

pub struct Permute;
impl<'a, B: Backend> Binary<'a, B> for Permute {
    fn forward(&self, a: &B::Storage<'a>, order: &B::Storage<'a>) -> B::Storage<'a> {
        let a_shape = a.shape().clone();
        a.permute(order)
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(a_shape))
    }

    fn backward(
        &self,
        ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
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
            .map(|u| B::Element::unsafe_from(*u))
            .collect();
        let inverse_order_td = <B::Storage<'a> as TensorData<B::Element>>::from_1d(&inverse_data);
        (
            d.permute(&inverse_order_td).unwrap_or(
                <B::Storage<'a> as TensorData<B::Element>>::ones(d.shape().clone()),
            ),
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
        )
    }

    fn tag(&self) -> &'static str {
        "permute"
    }
}

pub struct View;
impl<'a, B: Backend> Binary<'a, B> for View {
    fn forward(&self, lhs: &B::Storage<'a>, s: &B::Storage<'a>) -> B::Storage<'a> {
        assert!(lhs.is_contiguous(), "must be contiguous to view");
        let shape = Shape::new(s.collect().iter().map(|f| f.unsafe_to()).collect());
        lhs.reshape(shape)
    }

    fn backward(
        &self,
        ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        let shape = ctx
            .fst
            .as_ref()
            .map(|o| o.shape().clone())
            .unwrap_or(d.shape().clone());
        (
            d.reshape(shape),
            <B::Storage<'a> as TensorData<B::Element>>::from_scalar(B::Element::zero()),
        )
    }

    // TODO: id tag for gpu, improve
    fn tag(&self) -> &'static str {
        "id"
    }
}

pub struct MatMul;
impl<'a, B: Backend> Binary<'a, B> for MatMul {
    fn forward(&self, lhs: &B::Storage<'a>, rhs: &B::Storage<'a>) -> B::Storage<'a> {
        lhs.matmul(rhs)
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::zeros(
                lhs.shape().clone(),
            ))
    }

    fn backward(
        &self,
        ctx: &Context<B::Storage<'a>>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            ctx.snd
                .as_ref()
                .and_then(|b| b.transpose())
                .and_then(|b| d.matmul(&b))
                .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                    d.shape().clone(),
                )),
            ctx.fst
                .as_ref()
                .and_then(|a| a.transpose())
                .and_then(|a| a.matmul(d))
                .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                    d.shape().clone(),
                )),
        )
    }

    fn tag(&self) -> &'static str {
        "mm"
    }
}
