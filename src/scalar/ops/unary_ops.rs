pub struct Ln;
impl Unary<f64> for Ln {
    fn forward(&self, a: f64) -> f64 {
        if a <= 0. {
            0.
        } else {
            a.ln()
        }
    }

    fn backward(&self, ctx: &Context<f64>, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().filter(|v| **v != 0.).unwrap_or(&1.);
        d / a
    }
}

pub struct Inv;
impl Unary<f64> for Inv {
    fn forward(&self, a: f64) -> f64 {
        if a == 0. {
            0.
        } else {
            1. / a
        }
    }

    fn backward(&self, ctx: &Context<f64>, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().filter(|v| **v != 0.).unwrap_or(&1.);
        (-1. / (a.powf(2.))) * d
    }
}

pub struct Neg;
impl Unary<f64> for Neg {
    fn forward(&self, a: f64) -> f64 {
        -a
    }

    fn backward(&self, _ctx: &Context<f64>, d: f64) -> f64 {
        -d
    }
}

pub struct Sig;
impl Unary<f64> for Sig {
    fn forward(&self, a: f64) -> f64 {
        if a >= 0. {
            1. / (1. + (-a).exp())
        } else {
            a.exp() / (1. + a.exp())
        }
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context<f64>, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        let sig_a = self.forward(*a);
        sig_a * (1. - sig_a) * d
    }
}

pub struct Relu;
impl Unary<f64> for Relu {
    fn forward(&self, a: f64) -> f64 {
        a.max(0.)
    }

    fn backward(&self, ctx: &Context<f64>, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        if a > &0. {
            d
        } else {
            0.
        }
    }
}

pub struct Exp;
impl Unary<f64> for Exp {
    fn forward(&self, a: f64) -> f64 {
        a.exp()
    }

    fn backward(&self, ctx: &Context<f64>, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        a.exp() * d
    }
}

use proptest::prelude::*;

use crate::{autodiff::context::Context, function::unary::Unary};

proptest! {
    #[test]
    fn ln_tests(a in any::<f64>()) {
        let log = Ln {};
        let ctx = Context::default();
        let f = log.forward(a);
        if a <= 0. {
            assert_eq!(0., f);
        } else {
            assert_eq!(a.ln(), f);
        }
        let back = log.backward(&ctx.push(a), a);
        let exp = if a == 0. { 1. } else { a };
        assert_eq!(a / exp, back);
    }

    #[test]
    fn inv_tests(a in any::<f64>()) {
        let inv = Inv {};
        let ctx = Context::default();
        let f = inv.forward(a);
        if a == 0. {
            assert_eq!(0., f);
        } else {
            assert_eq!(1. / a, f);
        }
        let back = inv.backward(&ctx.push(a), a);
        let exp = if a == 0. { 1. } else { a };
        assert_eq!((-1. / (exp.powf(2.))) * a, back);
    }

    #[test]
    fn neg_tests(a in any::<f64>()) {
        let neg = Neg {};
        let ctx = Context::default();
        let f = neg.forward(a);
        assert_eq!(-a, f);
        let back = neg.backward(&ctx, a);
        assert_eq!(-a, back);
    }

    #[test]
    fn sig_tests(a in any::<f64>()) {
        let sig = Sig {};
        let ctx = Context::default();
        let f = sig.forward(a);
        if a >= 0. {
            assert_eq!(1. / (1. + (-a).exp()), f);
        } else {
            assert_eq!(a.exp() / (1. + a.exp()), f);
        }
        let back = sig.backward(&ctx.push(a), a);
        assert_eq!(f * (1. - f) * a, back);
    }

    #[test]
    fn relu_tests(a in any::<f64>()) {
        let relu = Relu {};
        let ctx = Context::default();
        let f = relu.forward(a);
        if a >= 0. {
            assert_eq!(a, f);
        } else {
            assert_eq!(0., f);
        }
        let back = relu.backward(&ctx.push(a), a);
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
        let f = exp.forward(a);
        assert_eq!(a.exp(), f);
        let back = exp.backward(&ctx.push(a), a);
        assert_eq!(a.exp() * a, back);
    }
}
