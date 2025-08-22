use crate::{
    autodiff::context::Context,
    backend::{backend::Backend, backend_type::BackendType},
    data::tensor_data::TensorData,
    math::element::Element,
};

use super::unary::Unary;

pub struct Neg;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Neg {
    fn forward(&self, a: &T) -> T {
        // TODO: make self.tag() work
        a.map(|e| -e, <Neg as Unary<E, BT, T>>::tag(self))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> T {
        d.map(|e| -e, <Neg as Unary<E, BT, T>>::tag(self))
    }

    fn tag(&self) -> &str {
        "neg"
    }
}

pub struct Inv;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Inv {
    fn forward(&self, a: &T) -> T {
        a.map(
            |e| {
                if e != E::zero() {
                    E::one() / e
                } else {
                    E::zero()
                }
            },
            <Inv as Unary<E, BT, T>>::tag(self),
        )
    }

    // todo: rm unwrap
    // decomposed for gpu, deriv 1/x => -1/x^2
    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        let a = ctx.fst.as_ref().unwrap();
        let a2 = a.zip(a, |e1, e2| e1 * e2, "mul").unwrap();
        let a2_inv = self.forward(&a2);
        let d_neg = d.map(|e| -e, "neg");
        d_neg
            .zip(&a2_inv, |e1, e2| e1 * e2, "mul")
            .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "inv"
    }
}

pub struct Ln;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Ln {
    fn forward(&self, a: &T) -> T {
        a.map(
            |e| if e > E::zero() { e.ln() } else { E::zero() },
            <Ln as Unary<E, BT, T>>::tag(self),
        )
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| {
                a.zip(
                    d,
                    |e1, e2| if e1 == E::zero() { e2 } else { e2 / e1 },
                    "inv",
                )
            })
            .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "ln"
    }
}

pub struct Sig;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Sig {
    fn forward(&self, a: &T) -> T {
        a.map(|e| e.sig(), <Sig as Unary<E, BT, T>>::tag(self))
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        //todo: rm unwrap
        let t = ctx.fst.as_ref().unwrap();
        let sig = self.forward(t);
        let minus_sig = sig.map(|e| -e, "neg");
        let one_minus_sig = <T as TensorData<E>>::from_scalar(E::one())
            .zip(&minus_sig, |e1, e2| e1 + e2, "add")
            .unwrap();
        let deriv = sig.zip(&one_minus_sig, |e1, e2| e1 * e2, "mul").unwrap();
        d.zip(&deriv, |e1, e2| e1 * e2, "mul").unwrap()
    }

    fn tag(&self) -> &str {
        "sig"
    }
}

// TODO: no out of the box support for relu, create a custom kernel
pub struct Relu;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Relu {
    fn forward(&self, a: &T) -> T {
        a.map(|e| e.relu(), <Relu as Unary<E, BT, T>>::tag(self))
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, |e1, e2| e1.relu_back(e2), "relu_back"))
            .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "relu"
    }
}

pub struct Exp;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Exp {
    fn forward(&self, a: &T) -> T {
        a.map(|e| e.exp(), <Exp as Unary<E, BT, T>>::tag(self))
    }

    // TODO: tag does not work for gpu, decompose
    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, |e1, e2| e1.exp_back(e2), "exp_back"))
            .unwrap_or(<T as TensorData<E>>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "exp"
    }
}

// aka make contiguous
pub struct Copy;
impl<E: Element, BT: BackendType, T: Backend<E, BT>> Unary<E, BT, T> for Copy {
    fn forward(&self, a: &T) -> T {
        a.map_broadcast(
            &<T as TensorData<E>>::zeros(a.shape().clone()),
            |f| f,
            <Copy as Unary<E, BT, T>>::tag(self),
        )
        .unwrap_or(a.map(|f| f, <Copy as Unary<E, BT, T>>::tag(self)))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> T {
        d.clone()
    }

    fn tag(&self) -> &str {
        "id"
    }
}
