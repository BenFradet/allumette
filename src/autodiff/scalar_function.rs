use super::context::Context;

// TODO: abstract over f64
trait Unary {
    // TODO: find a better encoding
    // need to have self otherwise can't be made into an object and can't dyn Unary
    fn forward(&self, ctx: Context, a: f64) -> f64;
    fn backward(&self, ctx: Context, d: f64) -> f64;
}

trait Binary {
    fn forward(&self, ctx: Context, a: f64, b: f64) -> f64;
    fn backward(&self, ctx: Context, d: f64) -> (f64, f64);
}

enum ScalarFunction {
    U(Box<dyn Unary>),
    B(Box<dyn Binary>),
}

impl ScalarFunction {
    fn add() -> Self { Self::B(Box::new(Add)) }

    fn log() -> Self { Self::U(Box::new(Log)) }

    fn mul() -> Self { Self::B(Box::new(Mul)) }

    fn inv() -> Self { Self::U(Box::new(Inv)) }

    fn neg() -> Self { Self::U(Box::new(Neg)) }

    fn sig() -> Self { Self::U(Box::new(Sig)) }

    fn relu() -> Self { Self::U(Box::new(Relu)) }

    fn exp() -> Self { Self::U(Box::new(Exp)) }

    fn lt() -> Self { Self::B(Box::new(Lt)) }

    fn eq() -> Self { Self::B(Box::new(Eq)) }
}

struct Add;
impl Binary for Add {
    fn forward(&self, _ctx: Context, a: f64, b: f64) -> f64 {
        a + b
    }

    fn backward(&self, _ctx: Context, d: f64) -> (f64, f64) {
        (d, d)
    }
}

struct Log;
impl Unary for Log {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        a.ln()
    }

    fn backward(&self, ctx: Context, d: f64) -> f64 {
        let vs = ctx.saved_values;
        let a = if vs.is_empty() { 0. } else { vs[0] };
        d / a
    }
}

struct Mul;
impl Binary for Mul {
    fn forward(&self, _ctx: Context, a: f64, b: f64) -> f64 {
        a * b
    }

    fn backward(&self, ctx: Context, d: f64) -> (f64, f64) {
        let vs = ctx.saved_values;
        let a = vs.get(0).unwrap_or(&1.);
        let b = vs.get(1).unwrap_or(&1.);
        (b * d, a * d)
    }
}

struct Inv;
impl Unary for Inv {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        1. / a
    }

    fn backward(&self, ctx: Context, d: f64) -> f64 {
        let vs = ctx.saved_values;
        let a = vs.get(0).unwrap_or(&0.);
        (- 1. / (a.powf(2.))) * d
    }
}

struct Neg;
impl Unary for Neg {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        -a
    }

    fn backward(&self, _ctx: Context, d: f64) -> f64 {
        -d
    }
}

struct Sig;
impl Unary for Sig {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        if a >= 0. {
            1. / (1. + (-a).exp())
        } else {
            a.exp() / (1. + a.exp())
        }
    }

    // sig'(x) = sig(x) * (1 - sig(x))
    fn backward(&self, ctx: Context, d: f64) -> f64 {
        let vs = ctx.saved_values;
        let a = vs.get(0).unwrap_or(&0.);
        let sig_a = self.forward(ctx, *a);
        sig_a * (1. - sig_a) * d
    }
}

struct Relu;
impl Unary for Relu {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        a.max(0.)
    }

    fn backward(&self, ctx: Context, d: f64) -> f64 {
        let vs = ctx.saved_values;
        let a = vs.get(0).unwrap_or(&0.);
        if a > &0. { d } else { 0. }
    }
}

struct Exp;
impl Unary for Exp {
    fn forward(&self, _ctx: Context, a: f64) -> f64 {
        a.exp()
    }

    fn backward(&self, ctx: Context, d: f64) -> f64 {
        let vs = ctx.saved_values;
        let a = vs.get(0).unwrap_or(&0.);
        a.exp() * d
    }
}

struct Lt;
impl Binary for Lt {
    fn forward(&self, _ctx: Context, a: f64, b: f64) -> f64 {
        if a < b { 1. } else { 0. }
    }

    fn backward(&self, _ctx: Context, _d: f64) -> (f64, f64) {
        (0., 0.)
    }
}

struct Eq;
impl Binary for Eq {
    fn forward(&self, _ctx: Context, a: f64, b: f64) -> f64 {
        if a == b { 1. } else { 0. }
    }

    fn backward(&self, _ctx: Context, _d: f64) -> (f64, f64) {
        (0., 0.)
    }
}