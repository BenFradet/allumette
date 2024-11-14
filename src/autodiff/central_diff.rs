// https://en.wikipedia.org/wiki/Finite_difference
// TODO: abstract unary binary
fn central_diff_u<F>(f: F, a: f64, eps: f64) -> f64 where F: Fn(f64) -> f64 {
    (f(a + eps) - f(a - eps)) / (2. * eps)
}

// TODO: find a better derivative indicator than arg
fn central_diff_b<F>(f: F, a: f64, b: f64, arg: bool, eps: f64) -> f64 where F: Fn(f64, f64) -> f64 {
    if arg {
        (f(a + eps, b) - f(a - eps, b)) / (2. * eps)
    } else {
        (f(a, b + eps) - f(a, b - eps)) / (2. * eps)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{add, exp, id, mul};

    use super::*;

    fn is_close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn central_diff_u_tests() -> () {
        assert!(is_close(1., central_diff_u(id, 5., 1e-6)));
        assert!(is_close(exp(2.), central_diff_u(exp, 2., 1e-6)));
    }

    #[test]
    fn central_diff_b_tests() -> () {
        assert!(is_close(1., central_diff_b(add, 5., 10., true, 1e-6)));
        assert!(is_close(10., central_diff_b(mul, 5., 10., true, 1e-6)));
        assert!(is_close(5., central_diff_b(mul, 5., 10., false, 1e-6)));
    }
}