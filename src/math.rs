pub fn mul(a: f64, b: f64) -> f64 {
    a * b
}

pub fn id(a: f64) -> f64 {
    a
}

pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

pub fn neg(a: f64) -> f64 {
    -a
}

pub fn lt(a: f64, b: f64) -> bool {
    a < b
}

pub fn eq(a: f64, b: f64) -> bool {
    a == b
}

pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

pub fn sigmoid(a: f64) -> f64 {
    if a >= 0. {
        1. / (1. + (-a).exp())
    } else {
        a.exp() / (1. + a.exp())
    }
}

pub fn relu(a: f64) -> f64 {
    a.max(0.)
}

pub fn relu_back(a: f64, b: f64) -> f64 {
    if a > 0. {
        b
    } else {
        0.
    }
}

pub fn log(a: f64) -> f64 {
    a.ln()
}

pub fn log_back(a: f64, b: f64) -> f64 {
    b / a
}

pub fn exp(a: f64) -> f64 {
    a.exp()
}

pub fn inv(a: f64) -> f64 {
    1. / a
}

pub fn inv_back(a: f64, b: f64) -> f64 {
    (- 1. / (a.powf(2.))) * b
}