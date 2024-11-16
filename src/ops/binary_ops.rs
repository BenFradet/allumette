use crate::autodiff::context::Context;

pub trait Binary {
    fn forward(&self, ctx: &Context, a: f64, b: f64) -> f64;
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
