use crate::{autodiff::context::Context, function::binary::Binary, math::math_binary};

pub struct Add;
impl Binary<f64, f64> for Add {
    fn forward(&self, a: f64, b: f64) -> f64 {
        math_binary::add(a, b)
    }

    fn backward(&self, _ctx: &Context<f64, f64>, d: f64) -> (f64, f64) {
        math_binary::add_back(d)
    }
}

pub struct Mul;
impl Binary<f64, f64> for Mul {
    fn forward(&self, a: f64, b: f64) -> f64 {
        math_binary::mul(a, b)
    }

    fn backward(&self, ctx: &Context<f64, f64>, d: f64) -> (f64, f64) {
        let a = ctx.a.unwrap_or(1.);
        let b = ctx.b.unwrap_or(1.);
        math_binary::mul_back(a, b, d)
    }
}

pub struct Div;
impl Binary<f64, f64> for Div {
    fn forward(&self, a: f64, b: f64) -> f64 {
        math_binary::div(a, b)
    }

    fn backward(&self, ctx: &Context<f64, f64>, d: f64) -> (f64, f64) {
        let a = ctx.a.unwrap_or(1.);
        let b = ctx.b.filter(|v| *v != 0.).unwrap_or(1.);
        math_binary::div_back(a, b, d)
    }
}

pub struct Lt;
impl Binary<f64, f64> for Lt {
    fn forward(&self, a: f64, b: f64) -> f64 {
        math_binary::lt(a, b)
    }

    fn backward(&self, _ctx: &Context<f64, f64>, _d: f64) -> (f64, f64) {
        (0., 0.)
    }
}

pub struct Eq;
impl Binary<f64, f64> for Eq {
    fn forward(&self, a: f64, b: f64) -> f64 {
        math_binary::eq(a, b)
    }

    fn backward(&self, _ctx: &Context<f64, f64>, _d: f64) -> (f64, f64) {
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
        let f = add.forward(a, b);
        assert_eq!(a + b, f);
        let back = add.backward(&ctx, a);
        assert_eq!((a, a), back);
    }

    #[test]
    fn mul_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let mul = Mul {};
        let ctx = Context::default();
        let f = mul.forward(a, b);
        if f.abs() == f64::INFINITY {
            assert_eq!((a * b).abs(), f64::INFINITY);
        } else {
            assert!(is_close(a * b, f));
        }
        let ctx2 = ctx.a(a).b(b);
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
        let f = div.forward(a, b);
        if b == 0. {
            assert_eq!(f, 0.);
        } else if f.abs() == f64::INFINITY {

            assert_eq!((a / b).abs(), f64::INFINITY);
        } else {
            assert!(is_close(a / b, f));
        }
        let ctx2 = ctx.a(a).b(b);
        let back = div.backward(&ctx2, d);
        if b == 0. {
            assert_eq!(d, back.0);
        } else if back.0.abs() == f64::INFINITY {
            assert_eq!((d / b).abs(), f64::INFINITY);
        } else {
            assert!(is_close(d / b, back.0));
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
        let f = lt.forward(a, b);
        if a < b { assert_eq!(1., f) } else { assert_eq!(0., f) }
        let back = lt.backward(&ctx, d);
        assert_eq!((0., 0.), back);
    }

    #[test]
    fn eq_tests(a in any::<f64>(), b in any::<f64>(), d in any::<f64>()) {
        let lt = Lt {};
        let ctx = Context::default();
        let f = lt.forward(a, b);
        assert!(f == 1. || f == 0.);
        let back = lt.backward(&ctx, d);
        assert_eq!((0., 0.), back);
    }
}
