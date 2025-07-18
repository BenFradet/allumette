pub fn ln(a: f32) -> f32 {
    if a <= 0. {
        0.
    } else {
        a.ln()
    }
}

pub fn ln_back(a: f32, d: f32) -> f32 {
    if a == 0. {
        d
    } else {
        d / a
    }
}

pub fn inv(a: f32) -> f32 {
    if a == 0. {
        0.
    } else {
        1. / a
    }
}

pub fn inv_back(a: f32, d: f32) -> f32 {
    if a == 0. {
        -d
    } else {
        -d / (a.powf(2.))
    }
}

pub fn neg(a: f32) -> f32 {
    -a
}

pub fn neg_back(d: f32) -> f32 {
    -d
}

pub fn sig(a: f32) -> f32 {
    if a >= 0. {
        1. / (1. + (-a).exp())
    } else {
        let a_exp = a.exp();
        a_exp / (1. + a_exp)
    }
}

// sig'(x) = sig(x) * (1 - sig(x))
pub fn sig_back(a: f32, d: f32) -> f32 {
    let sig_a = sig(a);
    sig_a * (1. - sig_a) * d
}

pub fn relu(a: f32) -> f32 {
    a.max(0.)
}

pub fn relu_back(a: f32, d: f32) -> f32 {
    if a > 0. {
        d
    } else {
        0.
    }
}

pub fn exp(a: f32) -> f32 {
    a.exp()
}

pub fn exp_back(a: f32, d: f32) -> f32 {
    a.exp() * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ln_tests(a in any::<f32>()) {
            let f = ln(a);
            if a <= 0. {
                assert_eq!(0., f);
            } else {
                assert_eq!(a.ln(), f);
            }
            let back = ln_back(a, a);
            let exp = if a == 0. { 1. } else { a };
            assert_eq!(a / exp, back);
        }

        #[test]
        fn inv_tests(a in any::<f32>()) {
            let f = inv(a);
            if a == 0. {
                assert_eq!(0., f);
            } else {
                assert_eq!(1. / a, f);
            }
            let back = inv_back(a, a);
            let exp = if a == 0. { 1. } else { a };
            assert_eq!(-a / (exp.powf(2.)), back);
        }

        #[test]
        fn neg_tests(a in any::<f32>()) {
            let f = neg(a);
            assert_eq!(-a, f);
            let back = neg_back(a);
            assert_eq!(-a, back);
        }

        #[test]
        fn sig_tests(a in any::<f32>()) {
            let f = sig(a);
            if a >= 0. {
                assert_eq!(1. / (1. + (-a).exp()), f);
            } else {
                assert_eq!(a.exp() / (1. + a.exp()), f);
            }
            let back = sig_back(a, a);
            assert_eq!(f * (1. - f) * a, back);
        }

        #[test]
        fn relu_tests(a in any::<f32>()) {
            let f = relu(a);
            if a >= 0. {
                assert_eq!(a, f);
            } else {
                assert_eq!(0., f);
            }
            let back = relu_back(a, a);
            if a > 0. {
                assert_eq!(a, back);
            } else {
                assert_eq!(0., back);
            }
        }

        #[test]
        fn exp_tests(a in any::<f32>()) {
            let f = exp(a);
            assert_eq!(a.exp(), f);
            let back = exp_back(a, a);
            assert_eq!(a.exp() * a, back);
        }
    }
}
