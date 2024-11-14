// TODO: abstract over f64
pub struct Context<'a> {
    grad: bool,
    saved_values: &'a[f64],
}

impl<'a> Default for Context<'a> {
    fn default() -> Self {
        Self {
            grad: true,
            saved_values: &[],
        }
    }
}

impl<'a> Context<'a> {
    fn new(no_grad: bool, values: &'a [f64]) -> Self {
        Self {
            grad: no_grad,
            saved_values: values,
        }
    }

    fn update(mut self, values: &'a [f64]) -> Self {
        if self.grad {
            self.saved_values = values;
            self
        } else {
            self
        }
    }

    fn grad(mut self) -> Self {
        self.grad = true;
        self
    }

    fn no_grad(mut self) -> Self {
        self.grad = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let a = [1., 2., 3.];
        let c = Context::new(false, &a);
        assert!(!c.grad);
        assert_eq!(&a, c.saved_values);
    }

    #[test]
    fn default_test() {
        let c = Context::default();
        assert!(c.grad);
        let exp: &[f64] = &[];
        assert_eq!(exp, c.saved_values);
    }

    #[test]
    fn update_test() {
        let a = [1., 2., 3.];
        let c = Context::new(false, &a);
        assert!(!c.grad);
        assert_eq!(&a, c.saved_values);
        let b = [2., 3., 4.];
        let c2 = c.update(&b);
        assert_eq!(&a, c2.saved_values);
        let c3 = c2.grad().update(&b);
        assert!(c3.grad);
        assert_eq!(&b, c3.saved_values);
    }
}