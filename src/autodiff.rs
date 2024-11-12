// https://en.wikipedia.org/wiki/Finite_difference
// TODO: abstract unary binary
fn central_diff_u<F>(f: F, a: f64, eps: f64) -> f64 where F: Fn(f64) -> f64 {
    (f(a + eps) - f(a - eps)) / 2. * eps
}

// TODO: find a better derivative indicator than arg
fn central_diff_b<F>(f: F, a: f64, b: f64, arg: bool, eps: f64) -> f64 where F: Fn(f64, f64) -> f64 {
    if arg {
        (f(a + eps, b) - f(a - eps, b)) / 2. * eps
    } else {
        (f(a, b + eps) - f(a, b - eps)) / 2. * eps
    }
}