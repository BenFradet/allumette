use crate::{
    autodiff::context::Context,
    backend::{backend::TensorBackend, backend_type::TensorBackendType},
    data::tensor_data::TensorData,
    math,
};

use super::unary::Unary;

pub struct Neg;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Neg {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::neg)
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> T {
        d.map(math::unary::neg_back)
    }

    fn tag(&self) -> &str {
        "neg"
    }
}

pub struct Inv;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Inv {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::inv)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::inv_back))
            .unwrap_or(<T as TensorData>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "inv"
    }
}

pub struct Ln;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Ln {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::ln)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::ln_back))
            .unwrap_or(<T as TensorData>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "ln"
    }
}

pub struct Sig;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Sig {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::sig)
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        //todo: rm unwrap
        let t = ctx.fst.as_ref().unwrap();
        let sig = self.forward(t);
        let minus_sig = sig.map(math::unary::neg);
        let one_minus_sig = <T as TensorData>::scalar(1.)
            .zip(&minus_sig, math::binary::add)
            .unwrap();
        let deriv = sig.zip(&one_minus_sig, math::binary::mul).unwrap();
        d.zip(&deriv, math::binary::mul).unwrap()
    }

    fn tag(&self) -> &str {
        "sig"
    }
}

pub struct Relu;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Relu {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::relu)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::relu_back))
            .unwrap_or(<T as TensorData>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "relu"
    }
}

pub struct Exp;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Exp {
    fn forward(&self, a: &T) -> T {
        a.map(math::unary::exp)
    }

    fn backward(&self, ctx: &Context<T>, d: &T) -> T {
        ctx.fst
            .as_ref()
            .and_then(|a| a.zip(d, math::unary::exp_back))
            .unwrap_or(<T as TensorData>::ones(d.shape().clone()))
    }

    fn tag(&self) -> &str {
        "exp"
    }
}

// make contiguous
pub struct Copy;
impl<BT: TensorBackendType, T: TensorBackend<BT>> Unary<BT, T> for Copy {
    fn forward(&self, a: &T) -> T {
        a.map_broadcast(&<T as TensorData>::zeros(a.shape().clone()), |f| f)
            .unwrap_or(a.map(|f| f))
    }

    fn backward(&self, _ctx: &Context<T>, d: &T) -> T {
        d.clone()
    }

    fn tag(&self) -> &str {
        "copy"
    }
}
