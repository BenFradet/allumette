use crate::backend::backend::Backend;

use crate::{
    backend::mode::Mode,
    math::element::Element,
    ops::ops::Ops,
    shaping::{order::Order, shape::Shape},
    storage::data::Data,
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

pub trait Binary<'a, B: Backend> {
    fn forward(&self, lhs: &B::Storage<'a>, rhs: &B::Storage<'a>) -> B::Storage<'a>;
    fn backward(
        &self,
        lhs: &B::Storage<'a>,
        rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>);
    fn tag(&self) -> &'static str;
}

pub struct Add;
impl<'a, B: Backend> Binary<'a, B> for Add {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(b, |e1, e2| e1 + e2, <Add as Binary<'a, B>>::tag(self))
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                a.shape().clone(),
            ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
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
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                a.shape().clone(),
            ))
    }

    fn backward(
        &self,
        lhs: &B::Storage<'a>,
        rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            rhs.zip(d, |e1, e2| e1 * e2, <Mul as Binary<'a, B>>::tag(self))
                .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                    d.shape().clone(),
                )),
            lhs.zip(d, |e1, e2| e1 * e2, <Mul as Binary<'a, B>>::tag(self))
                .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                    d.shape().clone(),
                )),
        )
    }

    fn tag(&self) -> &'static str {
        "mul"
    }
}

pub struct Div;
impl<'a, B: Backend> Binary<'a, B> for Div {
    fn forward(&self, a: &B::Storage<'a>, b: &B::Storage<'a>) -> B::Storage<'a> {
        a.zip(
            b,
            |e1, e2| {
                if e2 != B::Element::zero() {
                    e1 / e2
                } else {
                    B::Element::zero()
                }
            },
            <Div as Binary<'a, B>>::tag(self),
        )
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        lhs: &B::Storage<'a>,
        rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            // d / rhs
            rhs.zip(
                d,
                |er, ed| {
                    if er != B::Element::zero() {
                        ed / er
                    } else {
                        B::Element::zero()
                    }
                },
                "div_diff_lhs",
            )
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                d.shape().clone(),
            )),
            // -d * lhs / rhs^2
            lhs.zip(
                rhs,
                |el, er| {
                    if er != B::Element::zero() {
                        el / (er * er)
                    } else {
                        B::Element::zero()
                    }
                },
                "div_diff_rhs_1",
            )
            .and_then(|lr| lr.zip(d, |elr, ed| -ed * elr, "div_diff_rhs_2"))
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                d.shape().clone(),
            )),
        )
    }

    fn tag(&self) -> &'static str {
        "div"
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
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as Data<B::Element>>::zeros(d.shape().clone()),
            <B::Storage<'a> as Data<B::Element>>::zeros(d.shape().clone()),
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
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as Data<B::Element>>::zeros(d.shape().clone()),
            <B::Storage<'a> as Data<B::Element>>::zeros(d.shape().clone()),
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
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        _d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
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
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            d.clone(),
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
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
        .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
            a.shape().clone(),
        ))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        _d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
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
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(a_shape))
    }

    fn backward(
        &self,
        _lhs: &B::Storage<'a>,
        rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        let order = rhs.to_order();
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
        let inverse_order_td = <B::Storage<'a> as Data<B::Element>>::from_1d(&inverse_data);
        (
            d.permute(&inverse_order_td)
                .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                    d.shape().clone(),
                )),
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
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
        lhs: &B::Storage<'a>,
        _rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        let shape = lhs.shape().clone();
        (
            d.reshape(shape),
            <B::Storage<'a> as Data<B::Element>>::from_scalar(B::Element::zero()),
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
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::zeros(
                lhs.shape().clone(),
            ))
    }

    fn backward(
        &self,
        lhs: &B::Storage<'a>,
        rhs: &B::Storage<'a>,
        d: &B::Storage<'a>,
    ) -> (B::Storage<'a>, B::Storage<'a>) {
        (
            rhs.transpose()
                .and_then(|b| d.matmul(&b))
                .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                    d.shape().clone(),
                )),
            lhs.transpose()
                .and_then(|a| a.matmul(d))
                .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
                    d.shape().clone(),
                )),
        )
    }

    fn tag(&self) -> &'static str {
        "mm"
    }
}
