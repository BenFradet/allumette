use crate::autodiff::context::Context;

pub trait Binary {
    fn forward(&self, ctx: &Context, a: f64, b: f64) -> f64;
    // TODO: move to backward_a, backward_b
    fn backward(&self, ctx: &Context, d: f64) -> (f64, f64);
}

pub struct Add;
impl Binary for Add {
    fn forward(&self, _ctx: &Context, a: f64, b: f64) -> f64 {
        a + b
    }

    fn backward(&self, _ctx: &Context, d: f64) -> (f64, f64) {
        (d, d)
    }
}

pub struct Mul;
impl Binary for Mul {
    fn forward(&self, _ctx: &Context, a: f64, b: f64) -> f64 {
        a * b
    }

    fn backward(&self, ctx: &Context, d: f64) -> (f64, f64) {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&1.);
        let b = vs.get(1).unwrap_or(&1.);
        (b * d, a * d)
    }
}

pub struct Div;
impl Binary for Div {
    fn forward(&self, _ctx: &Context, a: f64, b: f64) -> f64 {
        if b == 0. {
            0.
        } else {
            a / b
        }
    }

    fn backward(&self, ctx: &Context, d: f64) -> (f64, f64) {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&1.);
        let b = vs.get(1).filter(|v| **v != 0.).unwrap_or(&1.);
        (d / b, a * d)
    }
}

pub struct Lt;
impl Binary for Lt {
    fn forward(&self, _ctx: &Context, a: f64, b: f64) -> f64 {
        if a < b {
            1.
        } else {
            0.
        }
    }

    fn backward(&self, _ctx: &Context, _d: f64) -> (f64, f64) {
        (0., 0.)
    }
}

pub struct Eq;
impl Binary for Eq {
    fn forward(&self, _ctx: &Context, a: f64, b: f64) -> f64 {
        if a == b {
            1.
        } else {
            0.
        }
    }

    fn backward(&self, _ctx: &Context, _d: f64) -> (f64, f64) {
        (0., 0.)
    }
}

use proptest::prelude::*;

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-4
}

proptest! {
    #[test]
    fn add_tests(a in any::<f64>(), b in any::<f64>()) {
        let add = Add {};
        let ctx = Context::default();
        let f = add.forward(&ctx, a, b);
        assert_eq!(a + b, f);
        let back = add.backward(&ctx, a);
        assert_eq!((a, a), back);
    }

    #[test]
    fn mul_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let mul = Mul {};
        let ctx = Context::default();
        let f = mul.forward(&ctx, a, b);
        if f.abs() == f64::INFINITY {
            assert_eq!((a * b).abs(), f64::INFINITY);
        } else {
            assert!(is_close(a * b, f));
        }
        let ctx2 = ctx.push(a).push(b);
        let back = mul.backward(&ctx2, d);
        if back.0.abs() == f64::INFINITY {
            assert_eq!((b * d).abs(), f64::INFINITY);
        } else {
            assert!(is_close(b * d, back.0));
        }
        if back.1.abs() == f64::INFINITY {
            assert_eq!((a * d).abs(), f64::INFINITY);
        } else {
            assert!(is_close(a * d, back.1));
        }
    }

    #[test]
    fn div_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let div = Div {};
        let ctx = Context::default();
        let f = div.forward(&ctx, a, b);
        if b == 0. {
            assert_eq!(f, 0.);
        } else {
            if f.abs() == f64::INFINITY {
                assert_eq!((a / b).abs(), f64::INFINITY);
            } else {
                assert!(is_close(a / b, f));
            }
        }
        let ctx2 = ctx.push(a).push(b);
        let back = div.backward(&ctx2, d);
        if b == 0. {
            assert_eq!(d, back.0);
        } else {
            if back.0.abs() == f64::INFINITY {
                assert_eq!((d / b).abs(), f64::INFINITY);
            } else {
                assert!(is_close(d / b, back.0));
            }
        }
        if back.1.abs() == f64::INFINITY {
            assert_eq!((a * d).abs(), f64::INFINITY);
        } else {
            assert!(is_close(a * d, back.1));
        }
    }

    #[test]
    fn lt_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let lt = Lt {};
        let ctx = Context::default();
        let f = lt.forward(&ctx, a, b);
        if a < b { assert_eq!(1., f) } else { assert_eq!(0., f) }
        let back = lt.backward(&ctx, d);
        assert_eq!((0., 0.), back);
    }

    #[test]
    fn eq_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let lt = Lt {};
        let ctx = Context::default();
        let f = lt.forward(&ctx, a, b);
        assert!(f == 1. || f == 0.);
        let back = lt.backward(&ctx, d);
        assert_eq!((0., 0.), back);
    }
}
