use crate::autodiff::context::Context;

// TODO: abstract over f64
pub trait Unary {
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, ctx: &Context, a: f64) -> f64;
    fn backward(&self, ctx: &Context, d: f64) -> f64;
}

pub struct Log;
impl Unary for Log {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        a.ln()
    }

    fn backward(&self, ctx: &Context, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = if vs.is_empty() { 0. } else { vs[0] };
        d / a
    }
}

pub struct Inv;
impl Unary for Inv {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        1. / a
    }

    fn backward(&self, ctx: &Context, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        (-1. / (a.powf(2.))) * d
    }
}

pub struct Neg;
impl Unary for Neg {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        -a
    }

    fn backward(&self, _ctx: &Context, d: f64) -> f64 {
        -d
    }
}

pub struct Sig;
impl Unary for Sig {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        if a >= 0. {
            1. / (1. + (-a).exp())
        } else {
            a.exp() / (1. + a.exp())
        }
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: &Context, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        let sig_a = self.forward(ctx, *a);
        sig_a * (1. - sig_a) * d
    }
}

pub struct Relu;
impl Unary for Relu {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        a.max(0.)
    }

    fn backward(&self, ctx: &Context, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        if a > &0. {
            d
        } else {
            0.
        }
    }
}

pub struct Exp;
impl Unary for Exp {
    fn forward(&self, _ctx: &Context, a: f64) -> f64 {
        a.exp()
    }

    fn backward(&self, ctx: &Context, d: f64) -> f64 {
        let vs = &ctx.saved_values;
        let a = vs.first().unwrap_or(&0.);
        a.exp() * d
    }
}
