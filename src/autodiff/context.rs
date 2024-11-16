// TODO: abstract over f64
// TODO: if no other use, convert saved_values to (v1, v2)
#[derive(Debug)]
pub struct Context {
    pub grad: bool,
    pub saved_values: Vec<f64>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            grad: true,
            saved_values: vec![],
        }
    }
}

impl Context {
    fn new(grad: bool, values: &[f64]) -> Self {
        Self {
            grad,
            saved_values: values.to_vec(),
        }
    }

    fn update(mut self, values: &[f64]) -> Self {
        if self.grad {
            self.saved_values = values.to_vec();
            self
        } else {
            self
        }
    }

    pub fn push(mut self, value: f64) -> Self {
        self.saved_values.push(value);
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
        self.saved_values.is_empty()
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
        assert_eq!(a.to_vec(), c.saved_values);
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
        assert_eq!(a.to_vec(), c.saved_values);
        let b = [2., 3., 4.];
        let c2 = c.update(&b);
        assert_eq!(a.to_vec(), c2.saved_values);
        let c3 = c2.grad().update(&b);
        assert!(c3.grad);
        assert_eq!(b.to_vec(), c3.saved_values);
    }

    #[test]
    fn push_test() {
        let a = [1., 2., 3.];
        let c = Context::new(false, &a);
        assert!(!c.grad);
        assert_eq!(a.to_vec(), c.saved_values);
        let c2 = c.push(5.);
        assert_eq!(vec![1., 2., 3., 5.], c2.saved_values);
    }
}
