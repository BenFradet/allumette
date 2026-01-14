use crate::{autodiff::context::Context, backend::backend::Backend};
use crate::{
    backend::mode::Mode, data::tensor_data::TensorData, math::element::Element,
    ops::tensor_ops::Ops,
};

pub trait Unary<'a, B: Backend> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a>;
    // TODO: remove ctx
    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a>;

    fn tag(&self) -> &'static str;
}

pub struct Neg;
impl<'a, B: Backend> Unary<'a, B> for Neg {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        // TODO: make self.tag() work
        a.map(|e| -e, <Neg as Unary<'a, B>>::tag(self))
    }

    fn backward(&self, _ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        d.map(|e| -e, <Neg as Unary<'a, B>>::tag(self))
    }

    fn tag(&self) -> &'static str {
        "neg"
    }
}

pub struct Inv;
impl<'a, B: Backend> Unary<'a, B> for Inv {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map(
            |e| {
                if e != B::Element::zero() {
                    B::Element::one() / e
                } else {
                    B::Element::zero()
                }
            },
            <Inv as Unary<'a, B>>::tag(self),
        )
    }

    // decomposed for gpu, deriv 1/x => -1/x^2
    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        let d_neg = d.map(|e| -e, "neg");
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(a, |e1, e2| e1 * e2, "mul"))
            .map(|a2| <Inv as Unary<'a, B>>::forward(self, &a2))
            .and_then(|a2_inv| d_neg.zip(&a2_inv, |e1, e2| e1 * e2, "mul"))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                d.shape().clone(),
            ))
    }

    fn tag(&self) -> &'static str {
        "inv"
    }
}

pub struct Ln;
impl<'a, B: Backend> Unary<'a, B> for Ln {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map(
            |e| {
                if e > B::Element::zero() {
                    e.ln()
                } else {
                    B::Element::zero()
                }
            },
            <Ln as Unary<'a, B>>::tag(self),
        )
    }

    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        ctx.fst
            .as_ref()
            .and_then(|a| {
                a.zip(
                    d,
                    |e1, e2| {
                        if e1 == B::Element::zero() {
                            e2
                        } else {
                            e2 / e1
                        }
                    },
                    "ln_diff",
                )
            })
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                d.shape().clone(),
            ))
    }

    fn tag(&self) -> &'static str {
        "ln"
    }
}

pub struct Sig;
impl<'a, B: Backend> Unary<'a, B> for Sig {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map(|e| e.sig(), <Sig as Unary<'a, B>>::tag(self))
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        ctx.fst
            .as_ref()
            .and_then(|t| {
                let sig = <Sig as Unary<'a, B>>::forward(self, t);
                let minus_sig = sig.map(|e| -e, "neg");
                let one_minus_sig = <B::Storage<'a> as TensorData<B::Element>>::from_scalar(
                    B::Element::one(),
                )
                .zip(&minus_sig, |e1, e2| e1 + e2, "add");
                one_minus_sig.and_then(|oms| sig.zip(&oms, |e1, e2| e1 * e2, "mul"))
            })
            .and_then(|deriv| d.zip(&deriv, |e1, e2| e1 * e2, "mul"))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                d.shape().clone(),
            ))
    }

    fn tag(&self) -> &'static str {
        "sig"
    }
}

pub struct Relu;
impl<'a, B: Backend> Unary<'a, B> for Relu {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map(|e| e.relu(), <Relu as Unary<'a, B>>::tag(self))
    }

    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, |e1, e2| e1.relu_diff(e2), "relu_diff"))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                d.shape().clone(),
            ))
    }

    fn tag(&self) -> &'static str {
        "relu"
    }
}

pub struct Exp;
impl<'a, B: Backend> Unary<'a, B> for Exp {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map(|e| e.exp(), <Exp as Unary<'a, B>>::tag(self))
    }

    fn backward(&self, ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        ctx.fst
            .as_ref()
            .map(|t| <Exp as Unary<'a, B>>::forward(self, t))
            .and_then(|exp| exp.zip(d, |e1, e2| e1 * e2, "mul"))
            .unwrap_or(<B::Storage<'a> as TensorData<B::Element>>::ones(
                d.shape().clone(),
            ))
    }

    fn tag(&self) -> &'static str {
        "exp"
    }
}

// aka make contiguous
pub struct Copy;
impl<'a, B: Backend> Unary<'a, B> for Copy {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        a.map_broadcast(
            &<B::Storage<'a> as TensorData<B::Element>>::zeros(a.shape().clone()),
            |f| f,
            <Copy as Unary<'a, B>>::tag(self),
        )
        .unwrap_or(a.map(|f| f, <Copy as Unary<'a, B>>::tag(self)))
    }

    fn backward(&self, _ctx: &Context<B::Storage<'a>>, d: &B::Storage<'a>) -> B::Storage<'a> {
        d.clone()
    }

    fn tag(&self) -> &'static str {
        "id"
    }
}
