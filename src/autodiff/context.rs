#[derive(Clone, Debug, PartialEq)]
pub struct Context<A, B> {
    pub grad: bool,
    pub a: Option<A>,
    pub b: Option<B>,
}

impl<A, B> Default for Context<A, B> {
    fn default() -> Self {
        Self {
            grad: true,
            a: None,
            b: None,
        }
    }
}

impl<A: Clone, B: Clone> Context<A, B> {
    fn new(grad: bool, a: A, b: B) -> Self {
        Self {
            grad,
            a: Some(a),
            b: Some(b),
        }
    }

    pub fn a(mut self, a: A) -> Self {
        self.a = Some(a);
        self
    }

    pub fn b(mut self, b: B) -> Self {
        self.b = Some(b);
        self
    }

    fn grad(mut self) -> Self {
        self.grad = true;
        self
    }

    fn no_grad(mut self) -> Self {
        self.grad = false;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_none() && self.b.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let c = Context::new(false, 1., 2.);
        assert!(!c.grad);
    }

    #[test]
    fn default_test() {
        let c: Context<f64, f64> = Context::default();
        assert!(c.grad);
        assert!(c.a.is_none());
        assert!(c.b.is_none());
    }
}
