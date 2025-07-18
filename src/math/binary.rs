pub fn add(a: f32, b: f32) -> f32 {
    a + b
}

pub fn add_back(d: f32) -> (f32, f32) {
    (d, d)
}

pub fn mul(a: f32, b: f32) -> f32 {
    a * b
}

pub fn div(a: f32, b: f32) -> f32 {
    if b == 0. {
        0.
    } else {
        a / b
    }
}

pub fn div_back(a: f32, b: f32, d: f32) -> (f32, f32) {
    if b == 0. {
        (d, a * d)
    } else {
        (d / b, a * d)
    }
}

pub fn lt(a: f32, b: f32) -> f32 {
    if a < b {
        1.
    } else {
        0.
    }
}

pub fn eq(a: f32, b: f32) -> f32 {
    if a == b {
        1.
    } else {
        0.
    }
}

pub fn is_close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-2
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn add_tests(a in any::<f32>(), b in any::<f32>()) {
        let f = add(a, b);
        assert_eq!(a + b, f);
        let back = add_back(a);
        assert_eq!((a, a), back);
    }

    #[test]
    fn mul_tests(a in any::<f32>(), b in any::<f32>()) {
        let f = mul(a, b);
        if f.abs() == f32::INFINITY {
            assert_eq!((a * b).abs(), f32::INFINITY);
        } else {
            assert!(is_close(a * b, f));
        }
    }

    #[test]
    fn div_tests(a in any::<f32>(), b in any::<f32>(), d in any::<f32>()) {
        let f = div(a, b);
        if b == 0. {
            assert_eq!(f, 0.);
        } else if f.abs() == f32::INFINITY {

            assert_eq!((a / b).abs(), f32::INFINITY);
        } else {
            assert!(is_close(a / b, f));
        }
        let back = div_back(a, b, d);
        if b == 0. {
            assert_eq!(d, back.0);
        } else if back.0.abs() == f32::INFINITY {
            assert_eq!((d / b).abs(), f32::INFINITY);
        } else {
            assert!(is_close(d / b, back.0));
        }
        if back.1.abs() == f32::INFINITY {
            assert_eq!((a * d).abs(), f32::INFINITY);
        } else {
            assert!(is_close(a * d, back.1));
        }
    }

    #[test]
    fn lt_tests(a in any::<f32>(), b in any::<f32>()) {
        let f = lt(a, b);
        if a < b { assert_eq!(1., f) } else { assert_eq!(0., f) }
    }

    #[test]
    fn eq_tests(a in any::<f32>(), b in any::<f32>()) {
        let f = eq(a, b);
        assert!(f == 1. || f == 0.);
    }
}
