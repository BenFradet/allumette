use crate::function::function::Function;
use proptest::{collection, prelude::*};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ops,
};

use super::{
    forward::Forward,
    ops::{
        binary_ops::{Add, All, Eq, IsClose, Lt, Mul, Permute, Sum, View},
        unary_ops::{Copy, Exp, Inv, Ln, Neg, Relu, Sig},
    },
    shaping::{order::Order, shape::Shape, strides::Strides},
    tensor_data::TensorData,
    tensor_history::TensorHistory,
};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: TensorData,
    pub grad: Option<Box<Tensor>>,
    pub history: TensorHistory,
    pub id: String,
}

impl Tensor {
    pub fn new(data: TensorData, history: TensorHistory) -> Self {
        let id = rand::thread_rng().gen::<u64>().to_string();
        Self {
            data,
            grad: None,
            history,
            id,
        }
    }

    pub fn from_data(data: TensorData) -> Self {
        let id = rand::thread_rng().gen::<u64>().to_string();
        Self {
            data,
            grad: None,
            history: TensorHistory::default(),
            id,
        }
    }

    pub fn scalar(data: f64) -> Self {
        Self::from_data(TensorData::scalar(data))
    }

    pub fn vec(data: Vec<f64>) -> Option<Self> {
        TensorData::vec(data).map(Self::from_data)
    }

    pub fn matrix(data: Vec<Vec<f64>>) -> Option<Self> {
        TensorData::matrix(data).map(Self::from_data)
    }

    pub fn backprop(&self, d: Tensor) -> HashMap<String, Self> {
        let sorted = self.topological_sort();
        let mut derivs = HashMap::from([(self.id.clone(), d)]);
        let mut res: HashMap<String, Self> = HashMap::new();
        for s in sorted {
            if let Some(current_deriv) = derivs.get(&s.id).cloned() {
                for (parent, grad) in s.chain_rule(current_deriv) {
                    let grad_tensor = Tensor::from_data(grad);
                    if parent.is_leaf() {
                        let new = match res.get(&parent.id) {
                            // TODO: remove clones
                            Some(s) => s.clone().accumulate_derivative(grad_tensor),
                            None => parent.clone().accumulate_derivative(grad_tensor),
                        };
                        res.insert(parent.id.clone(), new);
                    } else {
                        match derivs.remove(&parent.id) {
                            Some(e) => derivs.insert(parent.id.clone(), e + grad_tensor),
                            None => derivs.insert(parent.id.clone(), grad_tensor),
                        };
                    }
                }
            }
        }
        res
    }

    fn accumulate_derivative(mut self, d: Tensor) -> Self {
        if self.is_leaf() {
            self.grad = Some(Box::new(self.grad.map(|t| *t + d.clone()).unwrap_or(d)));
            self
        } else {
            self
        }
    }

    fn chain_rule(&self, d: Tensor) -> impl Iterator<Item = (&Self, TensorData)> {
        let derivatives = self
            .history
            .last_fn
            .as_ref()
            .map(|f| match f {
                Function::B(b) => {
                    let (da, db) = b.backward(&self.history.ctx, &d.data);
                    vec![da, db]
                }
                Function::U(u) => {
                    let da = u.backward(&self.history.ctx, &d.data);
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

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.data = self.data.reshape(shape);
        self
    }

    pub fn size(&self) -> usize {
        self.data.size()
    }

    pub fn item(&self) -> Option<f64> {
        if self.size() == 1 {
            Some(self.data.data[0])
        } else {
            None
        }
    }

    fn parents(&self) -> impl Iterator<Item = &Self> {
        self.history.inputs.iter()
    }

    fn is_leaf(&self) -> bool {
        self.history.last_fn.is_none()
    }

    pub fn lt(self, rhs: Tensor) -> Self {
        Forward::binary(Lt {}, self, rhs)
    }

    pub fn gt(self, rhs: Tensor) -> Self {
        Forward::binary(Lt {}, rhs, self)
    }

    pub fn eq(self, rhs: Tensor) -> Self {
        Forward::binary(Eq {}, self, rhs)
    }

    pub fn all(self, dim: usize) -> Self {
        Forward::binary(All {}, self, Tensor::scalar(dim as f64))
    }

    pub fn sum(self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => Forward::binary(Sum {}, self, Tensor::scalar(d as f64)),
            None => {
                let shape = Shape::scalar(self.size());
                let t = self.contiguous().view(shape).unwrap();
                Forward::binary(Sum {}, t, Tensor::scalar(0.))
            }
        }
    }

    pub fn mean(self, dim: usize) -> Self {
        self.sum(Some(dim)) / Self::from_data(TensorData::scalar(dim as f64))
    }

    pub fn permute(self, order: Order) -> Option<Self> {
        let fs = order.data.iter().map(|u| *u as f64).collect();
        Tensor::vec(fs).map(|td| Forward::binary(Permute {}, self, td))
    }

    pub fn view(self, shape: Shape) -> Option<Self> {
        let fs = shape.data().iter().map(|u| *u as f64).collect();
        Tensor::vec(fs).map(|td| Forward::binary(View {}, self, td))
    }

    pub fn contiguous(self) -> Self {
        Forward::unary(Copy {}, self)
    }

    pub fn is_close(self, rhs: Tensor) -> Self {
        Forward::binary(IsClose {}, self, rhs)
    }

    pub fn sigmoid(self) -> Self {
        Forward::unary(Sig {}, self)
    }

    pub fn relu(self) -> Self {
        Forward::unary(Relu {}, self)
    }

    pub fn ln(self) -> Self {
        Forward::unary(Ln {}, self)
    }

    pub fn exp(self) -> Self {
        Forward::unary(Exp {}, self)
    }

    pub fn inv(self) -> Self {
        Forward::unary(Inv {}, self)
    }

    pub fn arbitrary() -> impl Strategy<Value = Tensor> {
        TensorData::arbitrary().prop_map(Tensor::from_data)
    }

    pub fn arbitrary_tuple() -> impl Strategy<Value = (Tensor, Tensor)> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data1 = collection::vec(0.0f64..1., size);
                let data2 = collection::vec(0.0f64..1., size);
                (data1, data2, Just(shape))
            })
            .prop_map(|(data1, data2, shape)| {
                let strides: Strides = (&shape).into();
                (
                    Tensor::from_data(TensorData::new(data1, shape.clone(), strides.clone())),
                    Tensor::from_data(TensorData::new(data2, shape, strides)),
                )
            })
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Forward::binary(Add {}, self, rhs)
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs);
        Forward::binary(Add {}, self, new_rhs)
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Forward::binary(Mul {}, self, rhs)
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let new_rhs = Forward::unary(Inv {}, rhs);
        Forward::binary(Mul {}, self, new_rhs)
    }
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        math::{
            binary::{div, eq, is_close, lt},
            unary::{exp, inv, ln, relu, sig},
        },
        tensor::shaping::idx::Idx,
    };

    use super::*;

    fn unary_grad_central_difference<F>(tensor: Tensor, f: F, index: Idx) -> f64
    where
        F: Fn(Tensor) -> Tensor,
    {
        let eps = 1e-6;
        let shape = tensor.data.shape.clone();
        let up = Tensor::from_data(TensorData::epsilon(shape, index, eps));
        let add = tensor.clone() + up.clone();
        let sub = tensor - up;
        let delta = f(add).sum(None) - f(sub).sum(None);

        delta.item().unwrap_or(0.) / (2. * eps)
    }

    fn binary_grad_central_difference<F>(
        tensor1: Tensor,
        tensor2: Tensor,
        f: F,
        index: Idx,
        first: bool,
    ) -> f64
    where
        F: Fn(Tensor, Tensor) -> Tensor,
    {
        let eps = 1e-6;
        let shape = if first {
            tensor1.data.shape.clone()
        } else {
            tensor2.data.shape.clone()
        };
        let up = Tensor::from_data(TensorData::epsilon(shape, index, eps));
        let (add1, add2) = if first {
            (tensor1.clone() + up.clone(), tensor2.clone())
        } else {
            (tensor1.clone(), tensor2.clone() + up.clone())
        };
        let (sub1, sub2) = if first {
            (tensor1 - up, tensor2)
        } else {
            (tensor1, tensor2 - up)
        };
        let delta = f(add1, add2).sum(None) - f(sub1, sub2).sum(None);

        delta.item().unwrap_or(0.) / (2. * eps)
    }

    fn unary_assert<FT, FF>(t: Tensor, ft: FT, ff: FF)
    where
        FT: Fn(Tensor) -> Tensor,
        FF: Fn(f64) -> f64,
    {
        let data = t.data.clone();
        let res = ft(t);
        for idx in res.data.indices() {
            assert!(is_close(res.data[idx.clone()], ff(data[idx])));
        }
    }

    fn binary_assert<FT, FF>(t1: Tensor, t2: Tensor, ft: FT, ff: FF)
    where
        FT: Fn(Tensor, Tensor) -> Tensor,
        FF: Fn(f64, f64) -> f64,
    {
        let data1 = t1.data.clone();
        let data2 = t2.data.clone();
        let res = ft(t1, t2);
        for idx in res.data.indices() {
            assert!(is_close(
                res.data[idx.clone()],
                ff(data1[idx.clone()], data2[idx])
            ));
        }
    }

    proptest! {
        #[test]
        fn binary_tests((t1, t2) in Tensor::arbitrary_tuple()) {
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2, |f1, f2| f1 + f2);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2, |f1, f2| f1 - f2);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2, |f1, f2| f1 * f2);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 / t2, div);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1.gt(t2), |f1, f2| lt(f2, f1));
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1.lt(t2), lt);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1.eq(t2), eq);
        }

        #[test]
        fn unary_complex_test1(t in Tensor::arbitrary()) {
            let ft = |t: Tensor| (t.clone() + Tensor::scalar(100000.)).ln() + (t - Tensor::scalar(200.)).exp();
            let ff = |f| ln(f + 100000.) + exp(f - 200.);
            unary_assert(t.clone(), ft, ff);
        }

        #[test]
        fn unary_complex_test2(t in Tensor::arbitrary()) {
            let ft = |t: Tensor| (
                (
                    (
                        (
                            t * Tensor::scalar(10.) + Tensor::scalar(7.)
                        ).relu() * Tensor::scalar(6.) + Tensor::scalar(5.)
                    ).relu() * Tensor::scalar(10.)
                ).sigmoid()
            ).ln() / Tensor::scalar(50.);
            let ff = |f| ln(sig(relu(relu(f * 10. + 7.) * 6. + 5.) * 10.)) / 50.;
            unary_assert(t.clone(), ft, ff);
        }

        #[test]
        fn unary_tests(t in Tensor::arbitrary()) {
            unary_assert(t.clone(), |t| -t, |f| -f);
            unary_assert(t.clone(), |t| t.clone() * t, |f| f * f);
            unary_assert(t.clone(), |t| t.clone() * t.clone() * t, |f| f * f * f);
            unary_assert(t.clone(), |t| t.inv(), inv);
            unary_assert(t.clone(), |t| t.sigmoid(), sig);
            unary_assert(t.clone(), |t| t.ln(), ln);
            unary_assert(t.clone(), |t| t.relu(), relu);
            unary_assert(t.clone(), |t| t.exp(), exp);
        }
    }

    #[test]
    fn test_reduce_forward_one_dim() -> () {
        let shape = Shape::new(vec![3, 2]);
        let strides = (&shape).into();
        let td = TensorData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);
        let tensor = Tensor::from_data(td);
        let summed = tensor.sum(Some(0));

        let exp = Tensor::vec(vec![11., 16.]).unwrap();
        let is_close = summed.is_close(exp);
        let shape = Shape::scalar(is_close.size());
        assert_eq!(Some(1.), is_close.view(shape).unwrap().all(0).item());
    }

    #[test]
    fn test_reduce_forward_one_dim_2() -> () {
        let shape = Shape::new(vec![3, 2]);
        let strides = (&shape).into();
        let td = TensorData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);
        let tensor = Tensor::from_data(td);
        let summed = tensor.sum(Some(1));

        let exp =
            Tensor::from_data(TensorData::matrix(vec![vec![5.], vec![10.], vec![12.]]).unwrap());
        let is_close = summed.is_close(exp);
        let shape = Shape::new(vec![is_close.size()]);
        assert_eq!(Some(1.), is_close.view(shape).unwrap().all(0).item());
    }

    #[test]
    fn test_reduce_forward_all_dim() -> () {
        let shape = Shape::new(vec![3, 2]);
        let tensor = Tensor::vec(vec![2., 3., 4., 6., 5., 7.])
            .unwrap()
            .reshape(shape);
        let summed = tensor.sum(None);
        assert_eq!(Some(27.), summed.item());
    }
}
