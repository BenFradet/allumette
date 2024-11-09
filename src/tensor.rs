pub fn mul(a: f64, b: f64) -> f64 {
    a * b
}

fn id(a: f64) -> f64 {
    a
}

pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

pub fn neg(a: f64) -> f64 {
    -a
}

fn lt(a: f64, b: f64) -> bool {
    a < b
}

fn eq(a: f64, b: f64) -> bool {
    a == b
}

fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-2
}

fn sigmoid(a: f64) -> f64 {
    if a >= 0. {
        1. / (1. + (-a).exp())
    } else {
        a.exp() / (1. + a.exp())
    }
}

fn relu(a: f64) -> f64 {
    a.max(0.)
}

fn relu_back(a: f64, b: f64) -> f64 {
    if a > 0. {
        b
    } else {
        0.
    }
}

fn log(a: f64) -> f64 {
    a.ln()
}

fn log_back(a: f64, b: f64) -> f64 {
    b / a
}

fn exp(a: f64) -> f64 {
    a.exp()
}

fn inv(a: f64) -> f64 {
    1. / a
}

fn inv_back(a: f64, b: f64) -> f64 {
    (- 1. / (a.powf(2.))) * b
}