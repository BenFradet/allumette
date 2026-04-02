use crate::backend::backend::Backend;
use crate::{backend::mode::Mode, math::element::Element, ops::ops::Ops, storage::data::Data};

pub trait Unary<'a, B: Backend> {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a>;
    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a>;

    fn tag(&self) -> &'static str;
}

pub struct Neg;
impl<'a, B: Backend> Unary<'a, B> for Neg {
    fn forward(&self, a: &B::Storage<'a>) -> B::Storage<'a> {
        // TODO: make self.tag() work
        a.map(|e| -e, <Neg as Unary<'a, B>>::tag(self))
    }

    fn backward(&self, _input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
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

    // deriv 1/x => -1/x^2
    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        input
            .zip(
                d,
                |ei, ed| {
                    if ei == B::Element::zero() {
                        -ed
                    } else {
                        -ed / (ei * ei)
                    }
                },
                "inv_diff",
            )
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
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

    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        input
            .zip(
                d,
                |ei, ed| {
                    if ei == B::Element::zero() {
                        ed
                    } else {
                        ed / ei
                    }
                },
                "ln_diff",
            )
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
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
    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        input
            .zip(
                d,
                |ei, ed| ed * ei.sig() * (B::Element::one() - ei.sig()),
                "sig_diff",
            )
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
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

    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        input
            .zip(d, |ei, ed| ei.relu_diff(ed), "relu_diff")
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
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

    // TODO: fuse for gpu
    fn backward(&self, input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        let exp = <Exp as Unary<'a, B>>::forward(self, input);
        exp.zip(d, |ei, ed| ei * ed, "mul")
            .unwrap_or(<B::Storage<'a> as Data<B::Element>>::ones(
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
            &<B::Storage<'a> as Data<B::Element>>::zeros(a.shape().clone()),
            |f| f,
            <Copy as Unary<'a, B>>::tag(self),
        )
        .unwrap_or(a.map(|f| f, <Copy as Unary<'a, B>>::tag(self)))
    }

    fn backward(&self, _input: &B::Storage<'a>, d: &B::Storage<'a>) -> B::Storage<'a> {
        d.clone()
    }

    fn tag(&self) -> &'static str {
        "id"
    }
}
