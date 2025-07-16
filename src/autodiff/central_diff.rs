// https://en.wikipedia.org/wiki/Finite_difference
fn central_diff<F>(f: F, a: f64, eps: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(a + eps) - f(a - eps)) / (2. * eps)
}

fn central_diff_a<F>(f: F, a: f64, b: f64, eps: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    (f(a + eps, b) - f(a - eps, b)) / (2. * eps)
}

fn central_diff_b<F>(f: F, a: f64, b: f64, eps: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    (f(a, b + eps) - f(a, b - eps)) / (2. * eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn central_diff_u_tests() {
        assert!(is_close(1., central_diff(|a| a, 5., 1e-6)));
        assert!(is_close(2f64.exp(), central_diff(|a| a.exp(), 2., 1e-6)));
    }

    #[test]
    fn central_diff_a_tests() {
        assert!(is_close(1., central_diff_a(|a, b| a + b, 5., 10., 1e-6)));
        assert!(is_close(10., central_diff_a(|a, b| a * b, 5., 10., 1e-6)));
    }

    #[test]
    fn central_diff_b_tests() {
        assert!(is_close(5., central_diff_b(|a, b| a * b, 5., 10., 1e-6)));
    }
}
