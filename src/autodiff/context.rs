#[derive(Clone, Debug, PartialEq)]
pub struct Context<A> {
    pub grad: bool,
    pub fst: Option<A>,
    pub snd: Option<A>,
}

impl<A> Default for Context<A> {
    fn default() -> Self {
        Self {
            grad: true,
            fst: None,
            snd: None,
        }
    }
}

impl<A: Clone> Context<A> {
    pub fn fst(mut self, a: A) -> Self {
        self.fst = Some(a);
        self
    }

    pub fn snd(mut self, a: A) -> Self {
        self.snd = Some(a);
        self
    }

    pub fn grad(mut self) -> Self {
        self.grad = true;
        self
    }

    pub fn no_grad(mut self) -> Self {
        self.grad = false;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.fst.is_none() && self.snd.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_test() {
        let c: Context<f64> = Context::default();
        assert!(c.grad);
        assert!(c.fst.is_none());
        assert!(c.snd.is_none());
    }
}
