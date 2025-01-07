use std::{
    collections::{HashMap, HashSet, VecDeque},
    ops,
};

use rand::{thread_rng, Rng};

use super::{autodiff::forward::Forward, ops::{binary_ops::{Add, Div, Eq, Lt, Mul}, unary_ops::{Exp, Ln, Neg, Relu, Sig}}, scalar_function::ScalarFunction, scalar_history::ScalarHistory};

// TODO: abstract over f64
#[derive(Clone, Debug)]
pub struct Scalar {
    pub v: f64,
    pub derivative: Option<f64>,
    pub history: ScalarHistory,
    pub id: String,
}

impl Scalar {
    pub fn new(v: f64) -> Self {
        let mut rng = thread_rng();
        let id: u64 = rng.gen();
        Self {
            v,
            derivative: None,
            history: ScalarHistory::default(),
            id: id.to_string(),
        }
    }

    pub fn history(mut self, history: ScalarHistory) -> Self {
        self.history = history;
        self
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

    pub fn derivative(mut self, d: Option<f64>) -> Self {
        self.derivative = d;
        self
    }

    fn is_constant(&self) -> bool {
        self.history.is_empty()
    }

    fn is_leaf(&self) -> bool {
        self.history.last_fn.is_none()
    }

    fn parents(&self) -> impl Iterator<Item = &Self> {
        self.history.inputs.iter()
    }

    fn accumulate_derivative(mut self, d: f64) -> Self {
        if self.is_leaf() {
            self.derivative = Some(self.derivative.unwrap_or(0.) + d);
            self
        } else {
            self
        }
    }

    fn chain_rule(&self, d: f64) -> impl Iterator<Item = (&Self, f64)> {
        let derivatives = self
            .history
            .last_fn
            .as_ref()
            .map(|f| match f {
                ScalarFunction::B(b) => {
                    let (da, db) = b.backward(&self.history.ctx, d);
                    vec![da, db]
                }
                ScalarFunction::U(u) => {
                    let da = u.backward(&self.history.ctx, d);
                    vec![da]
                }
            })
            .unwrap_or_default();
        let inputs = &self.history.inputs;
        inputs.iter().zip(derivatives)
    }

    fn topological_sort(&self) -> impl Iterator<Item = &Self> {
        let mut queue = VecDeque::new();
        queue.push_back(self);
        let mut visited: HashSet<String> = HashSet::from([self.id.clone()]);
        let mut result = Vec::new();
        while let Some(var) = queue.pop_front() {
            for parent in var.parents() {
                if !visited.contains(&parent.id) {
                    visited.insert(parent.id.clone());
                    queue.push_back(parent);
                }
            }
            result.push(var);
        }
        result.into_iter()
    }

    pub fn backprop(&self, d: f64) -> HashMap<String, Self> {
        let sorted = self.topological_sort();
        let mut derivs = HashMap::from([(self.id.clone(), d)]);
        let mut res: HashMap<String, Scalar> = HashMap::new();
        for s in sorted {
            if let Some(current_deriv) = derivs.get(&s.id).cloned() {
                for (parent, grad) in s.chain_rule(current_deriv) {
                    if parent.is_leaf() {
                        let new = match res.get(&parent.id) {
                            // TODO: remove clones
                            Some(s) => s.clone().accumulate_derivative(grad),
                            None => parent.clone().accumulate_derivative(grad),
                        };
                        res.insert(parent.id.clone(), new);
                    } else {
                        match derivs.get_mut(&parent.id) {
                            Some(e) => *e += grad,
                            None => {
                                derivs.insert(parent.id.clone(), grad);
                            }
                        }
                    }
                }
            }
        }
        res
    }

    pub fn ln(&self) -> Self {
        Forward::unary(Ln {}, self)
    }

    pub fn exp(&self) -> Self {
        Forward::unary(Exp {}, self)
    }

    pub fn sig(&self) -> Self {
        Forward::unary(Sig {}, self)
    }

    pub fn relu(&self) -> Self {
        Forward::unary(Relu {}, self)
    }

    pub fn eq(&self, rhs: &Scalar) -> Self {
        Forward::binary(Eq {}, self, rhs)
    }

    pub fn lt(&self, rhs: &Scalar) -> Self {
        Forward::binary(Lt {}, self, rhs)
    }

    pub fn gt(&self, rhs: &Scalar) -> Self {
        Forward::binary(Lt {}, rhs, self)
    }
}

impl ops::Add<&Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        Forward::binary(Add {}, &self, rhs)
    }
}

impl ops::Add<Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        Forward::binary(Add {}, &self, &rhs)
    }
}

impl ops::Add<Scalar> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        Forward::binary(Add {}, self, &rhs)
    }
}

impl ops::Add<&Scalar> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        Forward::binary(Add {}, self, rhs)
    }
}

impl ops::Mul<&Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        Forward::binary(Mul {}, &self, rhs)
    }
}

impl ops::Mul<&Scalar> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        Forward::binary(Mul {}, self, rhs)
    }
}

impl ops::Div<&Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &Scalar) -> Self::Output {
        Forward::binary(Div {}, &self, rhs)
    }
}

impl ops::Sub<&Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs);
        Forward::binary(Add {}, &self, &new_rhs)
    }
}

impl ops::Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, &self)
    }
}

impl ops::Neg for &Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::scalar::{autodiff::context::Context, ops::binary_ops::Binary, scalar_function::ScalarFunction, scalar_history::ScalarHistory};

    struct F1;
    impl Binary for F1 {
        fn forward(&self, a: f64, b: f64) -> f64 {
            a + b + 10.
        }

        fn backward(&self, _ctx: &Context, d: f64) -> (f64, f64) {
            (d, d)
        }
    }

    struct F2;
    impl Binary for F2 {
        fn forward(&self, a: f64, b: f64) -> f64 {
            a * b + a
        }

        fn backward(&self, ctx: &Context, d: f64) -> (f64, f64) {
            let vs = &ctx.saved_values;
            let a = vs.first().unwrap_or(&1.);
            let b = vs.get(1).unwrap_or(&1.);
            (d * (b + 1.), d * a)
        }
    }

    #[test]
    fn backprop_test1() -> () {
        let var = Scalar::new(0.);
        let var_id = var.id.to_owned();
        let var2 = Forward::binary(F1 {}, &Scalar::new(0.), &var);
        let backprop = var2.backprop(5.);
        let res = backprop.get(&var_id);
        assert_eq!(res.and_then(|s| s.derivative), Some(5.));
    }

    #[test]
    fn backprop_test2() -> () {
        let var = Scalar::new(0.);
        let var_id = var.id.to_owned();
        let var2 = Forward::binary(F1 {}, &Scalar::new(0.), &var);
        let var3 = Forward::binary(F1 {}, &Scalar::new(0.), &var2);
        let backprop = var3.backprop(5.);
        let res = backprop.get(&var_id);
        assert_eq!(res.and_then(|s| s.derivative), Some(5.));
    }

    #[test]
    fn backprop_test3() -> () {
        let var = Scalar::new(0.);
        let var_id = var.id.to_owned();
        let var2 = Forward::binary(F1 {}, &Scalar::new(0.), &var);
        let var3 = Forward::binary(F1 {}, &Scalar::new(0.), &var);
        let var4 = Forward::binary(F1 {}, &var2, &var3);
        let backprop = var4.backprop(5.);
        let res = backprop.get(&var_id);
        assert_eq!(res.and_then(|s| s.derivative), Some(10.));
    }

    #[test]
    fn backprop_test4() -> () {
        let var = Scalar::new(0.);
        let var_id = var.id.to_owned();
        let var1 = Forward::binary(F1 {}, &Scalar::new(0.), &var);
        let var2 = Forward::binary(F1 {}, &Scalar::new(0.), &var1);
        let var3 = Forward::binary(F1 {}, &Scalar::new(0.), &var1);
        let var4 = Forward::binary(F1 {}, &var2, &var3);
        let backprop = var4.backprop(5.);
        let res = backprop.get(&var_id);
        assert_eq!(res.and_then(|s| s.derivative), Some(10.));
    }

    #[test]
    fn topological_sort_test2() -> () {
        let x = Scalar::new(1.).id(0.to_string());
        let y = Scalar::new(2.).id(1.to_string());
        let log_z = Scalar::new(3.).id(10.to_string()).history(
            ScalarHistory::default()
                .push_input(x.clone())
                .push_input(y.clone()),
        );
        let exp_z = Scalar::new(4.).id(11.to_string()).history(
            ScalarHistory::default()
                .push_input(x.clone())
                .push_input(y.clone()),
        );
        let h = Scalar::new(5.)
            .id(100.to_string())
            .history(ScalarHistory::default().push_input(log_z).push_input(exp_z));
        let sorted: Vec<_> = h.topological_sort().map(|s| s.id.to_owned()).collect();
        assert_eq!(
            vec![
                100.to_string(),
                10.to_string(),
                11.to_string(),
                0.to_string(),
                1.to_string()
            ],
            sorted
        );
    }

    #[test]
    fn topological_sort_test1() -> () {
        let five = Scalar::new(5.).id(5.to_string());
        let four = Scalar::new(4.).id(4.to_string());
        let z = Scalar::new(0.).id(0.to_string()).history(
            ScalarHistory::default()
                .push_input(four.clone())
                .push_input(five.clone()),
        );
        let two = Scalar::new(2.)
            .id(2.to_string())
            .history(ScalarHistory::default().push_input(five));
        let three = Scalar::new(3.)
            .id(3.to_string())
            .history(ScalarHistory::default().push_input(two));
        let one = Scalar::new(1.)
            .id(1.to_string())
            .history(ScalarHistory::default().push_input(four).push_input(three));
        let sorted_z: Vec<_> = z.topological_sort().map(|s| s.id.to_owned()).collect();
        assert_eq!(vec![0.to_string(), 4.to_string(), 5.to_string()], sorted_z);
        let sorted_one: Vec<_> = one.topological_sort().map(|s| s.id.to_owned()).collect();
        assert_eq!(
            vec![
                1.to_string(),
                4.to_string(),
                3.to_string(),
                2.to_string(),
                5.to_string()
            ],
            sorted_one
        );
    }

    #[test]
    fn chain_rule_test1() -> () {
        let hist = ScalarHistory::default()
            .last_fn(ScalarFunction::B(Rc::new(F1 {})))
            .push_input(Scalar::new(0.))
            .push_input(Scalar::new(0.));
        let constant = Scalar::new(0.).history(hist);
        let back: Vec<_> = constant.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((_, deriv)) = back.first() {
            assert_eq!(5., *deriv);
        } else {
            panic!("test failure")
        }
    }

    #[test]
    fn chain_rule_test2() -> () {
        let f2 = F2 {};
        let y = Forward::binary(f2, &Scalar::new(10.), &Scalar::new(5.));
        let back: Vec<_> = y.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((_, deriv)) = back.get(1) {
            assert_eq!(50., *deriv);
        } else {
            panic!("test failure")
        }
    }

    #[test]
    fn chain_rule_test3() -> () {
        let f2 = F2 {};
        let v1 = Scalar::new(5.);
        let v1_id = v1.id.to_owned();
        let v2 = Scalar::new(10.);
        let v2_id = v2.id.to_owned();
        let y = Forward::binary(f2, &v1, &v2);
        let back: Vec<_> = y.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((v, deriv)) = back.first() {
            assert_eq!(v1_id, v.id);
            assert_eq!(55., *deriv);
        } else {
            panic!("test failure")
        }
        if let Some((v, deriv)) = back.get(1) {
            assert_eq!(v2_id, v.id);
            assert_eq!(25., *deriv);
        } else {
            panic!("test failure")
        }
    }
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn add_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) + Scalar::new(b);
        assert_eq!(a + b, res.v);
    }

    #[test]
    fn mul_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) * &Scalar::new(b);
        assert_eq!(a * b, res.v);
    }

    #[test]
    fn div_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) / &Scalar::new(b);
        if b == 0. {
            assert_eq!(0., res.v);
        } else {
            assert_eq!(a / b, res.v);
        }
    }

    #[test]
    fn sub_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) - &Scalar::new(b);
        assert_eq!(a - b, res.v);
    }

    #[test]
    fn neg_test(a in any::<f64>()) {
        let res = - Scalar::new(a);
        assert_eq!(-a, res.v);
    }

    #[test]
    fn ln_test(a in any::<f64>()) {
        let res = Scalar::new(a).ln();
        if a <= 0. {
            assert_eq!(0., res.v);
        } else {
            assert_eq!(a.ln(), res.v);
        }
    }

    #[test]
    fn exp_test(a in any::<f64>()) {
        let res = Scalar::new(a).exp();
        assert_eq!(a.exp(), res.v);
    }

    #[test]
    fn sig_test(a in any::<f64>()) {
        let res = Scalar::new(a).sig();
        if a >= 0. {
            assert_eq!(1. / (1. + (-a).exp()), res.v);
        } else {
            assert_eq!(a.exp() / (1. + a.exp()), res.v);
        }
    }

    #[test]
    fn relu_test(a in any::<f64>()) {
        let res = Scalar::new(a).relu();
        assert_eq!(a.max(0.), res.v);
    }

    #[test]
    fn eq_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).eq(&Scalar::new(b));
        if a == b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }

    #[test]
    fn lt_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).lt(&Scalar::new(b));
        if a < b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }

    #[test]
    fn gt_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).gt(&Scalar::new(b));
        if a > b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }
}
