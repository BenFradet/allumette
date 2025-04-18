use crate::{autodiff::context::Context, function::unary::Unary, math};

pub struct Ln;
impl Unary<f64> for Ln {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::ln(*a)
    }

    fn backward(&self, ctx: &Context<f64>, d: &f64) -> f64 {
        let a = ctx.fst.filter(|v| *v != 0.).unwrap_or(1.);
        math::unary::ln_back(a, *d)
    }

    fn tag(&self) -> &str {
        "ln"
    }
}

pub struct Inv;
impl Unary<f64> for Inv {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::inv(*a)
    }

    fn backward(&self, ctx: &Context<f64>, d: &f64) -> f64 {
        let a = ctx.fst.filter(|v| *v != 0.).unwrap_or(1.);
        math::unary::inv_back(a, *d)
    }

    fn tag(&self) -> &str {
        "inv"
    }
}

pub struct Neg;
impl Unary<f64> for Neg {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::neg(*a)
    }

    fn backward(&self, _ctx: &Context<f64>, d: &f64) -> f64 {
        math::unary::neg_back(*d)
    }

    fn tag(&self) -> &str {
        "neg"
    }
}

pub struct Sig;
impl Unary<f64> for Sig {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::sig(*a)
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<f64>, d: &f64) -> f64 {
        let a = ctx.fst.unwrap_or(0.);
        math::unary::sig_back(a, *d)
    }

    fn tag(&self) -> &str {
        "sig"
    }
}

pub struct Relu;
impl Unary<f64> for Relu {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::relu(*a)
    }

    fn backward(&self, ctx: &Context<f64>, d: &f64) -> f64 {
        let a = ctx.fst.unwrap_or(0.);
        math::unary::relu_back(a, *d)
    }

    fn tag(&self) -> &str {
        "relu"
    }
}

pub struct Exp;
impl Unary<f64> for Exp {
    fn forward(&self, a: &f64) -> f64 {
        math::unary::exp(*a)
    }

    fn backward(&self, ctx: &Context<f64>, d: &f64) -> f64 {
        let a = ctx.fst.unwrap_or(0.);
        math::unary::exp_back(a, *d)
    }

    fn tag(&self) -> &str {
        "exp"
    }
}

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-4
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn ln_tests(a in any::<f64>()) {
        let log = Ln {};
        let ctx = Context::default();
        let f = log.forward(&a);
        if a <= 0. {
            assert_eq!(0., f);
        } else {
            assert_eq!(a.ln(), f);
        }
        let back = log.backward(&ctx.fst(a), &a);
        let exp = if a == 0. { 1. } else { a };
        assert_eq!(a / exp, back);
    }

    #[test]
    fn inv_tests(a in any::<f64>()) {
        let inv = Inv {};
        let ctx = Context::default();
        let f = inv.forward(&a);
        if a == 0. {
            assert_eq!(0., f);
        } else {
            assert_eq!(1. / a, f);
        }
        let back = inv.backward(&ctx.fst(a), &a);
        let exp = if a == 0. { 1. } else { a };
        assert_eq!(-a / (exp.powf(2.)), back);
    }

    #[test]
    fn neg_tests(a in any::<f64>()) {
        let neg = Neg {};
        let ctx = Context::default();
        let f = neg.forward(&a);
        assert_eq!(-a, f);
        let back = neg.backward(&ctx, &a);
        assert_eq!(-a, back);
    }

    #[test]
    fn sig_tests(a in any::<f64>()) {
        let sig = Sig {};
        let ctx = Context::default();
        let f = sig.forward(&a);
        if a >= 0. {
            assert_eq!(1. / (1. + (-a).exp()), f);
        } else {
            assert_eq!(a.exp() / (1. + a.exp()), f);
        }
        let back = sig.backward(&ctx.fst(a), &a);
        assert_eq!(f * (1. - f) * a, back);
    }

    #[test]
    fn relu_tests(a in any::<f64>()) {
        let relu = Relu {};
        let ctx = Context::default();
        let f = relu.forward(&a);
        if a >= 0. {
            assert_eq!(a, f);
        } else {
            assert_eq!(0., f);
        }
        let back = relu.backward(&ctx.fst(a), &a);
        if a > 0. {
            assert_eq!(a, back);
        } else {
            assert_eq!(0., back);
        }
    }

    #[test]
    fn exp_tests(a in any::<f64>()) {
        let exp = Exp {};
        let ctx = Context::default();
        let f = exp.forward(&a);
        assert_eq!(a.exp(), f);
        let back = exp.backward(&ctx.fst(a), &a);
        assert_eq!(a.exp() * a, back);
    }
}
