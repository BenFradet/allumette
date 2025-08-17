use crate::{
    autodiff::{forward::Forward, history::History},
    backend::{backend::Backend, backend_type::BackendType},
    data::{cpu_tensor_data::CpuTensorData, tensor_data::TensorData},
    math::element::Element,
    ops::{
        binary_ops::{Add, All, Eq, IsClose, Lt, MatMul, Mul, Permute, Sum, View},
        function::Function,
        unary_ops::{Copy, Exp, Inv, Ln, Neg, Relu, Sig},
    },
    shaping::{order::Order, shape::Shape, strides::Strides},
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};
use proptest::{collection, prelude::*};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ops,
};

#[derive(Clone, Debug)]
pub struct Tensor<E: Element, BT: BackendType, T: Backend<E, BT>> {
    pub data: T,
    pub grad: Option<Box<Tensor<E, BT, T>>>,
    pub history: History<E, BT, T>,
    pub id: String,
    pub is_constant: bool,
}

//static TENSOR_COUNT: AtomicU32 = AtomicU32::new(0);

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Tensor<E, BT, T>
where
    E: Element,
    BT: std::fmt::Debug + Clone,
    T: TensorData<E> + std::fmt::Debug + Clone,
{
    pub fn new(data: T, history: History<E, BT, T>) -> Self {
        let id = rand::thread_rng().gen::<u64>().to_string();
        //let id = TENSOR_COUNT.fetch_add(1, Ordering::Relaxed);
        Self {
            data,
            grad: None,
            history,
            id: id.to_string(),
            is_constant: false,
        }
    }

    pub fn from_data(data: T) -> Self {
        let id = rand::thread_rng().gen::<u64>().to_string();
        Self {
            data,
            grad: None,
            history: History::default(),
            id,
            is_constant: false,
        }
    }

    pub fn from_scalar(data: E) -> Self {
        Self::from_data(<T as TensorData<E>>::from_scalar(data)).make_constant()
    }

    pub fn from_1d(data: &[E]) -> Self {
        Self::from_data(<T as TensorData<E>>::from_1d(data))
    }

    pub fn from_2d(data: &[&[E]]) -> Option<Self> {
        <T as TensorData<E>>::from_2d(data).map(Self::from_data)
    }

    pub fn history(mut self, h: History<E, BT, T>) -> Self {
        self.history = h;
        self
    }

    pub fn grad(mut self, grad: Option<Tensor<E, BT, T>>) -> Self {
        self.grad = grad.map(Box::new);
        self
    }

    pub fn data(mut self, data: T) -> Self {
        self.data = data;
        self
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

    fn make_constant(mut self) -> Self {
        self.is_constant = true;
        self
    }

    pub fn backward(&self) -> HashMap<String, Self> {
        assert!(
            *self.data.shape() == Shape::new(vec![1]),
            "use backprop for non-scalar tensors"
        );
        self.backprop(Self::from_scalar(E::one()))
    }

    pub fn backprop(&self, d: Tensor<E, BT, T>) -> HashMap<String, Self> {
        let sorted = self.topological_sort_dfs();
        let mut derivs = HashMap::from([(&self.id, d)]);
        let mut res: HashMap<String, Self> = HashMap::new();
        for s in sorted {
            if let Some(current_deriv) = derivs.get(&s.id).cloned() {
                for (parent, grad) in s.chain_rule(&current_deriv.data) {
                    let grad_tensor = Tensor::from_data(grad).make_constant();
                    if parent.is_leaf() {
                        let new = match res.get(&parent.id) {
                            // TODO: remove clones
                            Some(s) => s.clone().accumulate_derivative(grad_tensor),
                            None => parent.clone().accumulate_derivative(grad_tensor),
                        };
                        res.insert(parent.id.clone(), new);
                    } else {
                        match derivs.remove(&parent.id) {
                            Some(e) => derivs.insert(&parent.id, e + grad_tensor),
                            None => derivs.insert(&parent.id, grad_tensor),
                        };
                    }
                }
            }
        }
        res
    }

    fn accumulate_derivative(mut self, d: Tensor<E, BT, T>) -> Self {
        if self.is_leaf() {
            let grad = self.grad.map(|t| *t + d.clone()).unwrap_or(d);
            self.grad = Some(Box::new(grad.clone()));
            self
        } else {
            self
        }
    }

    fn chain_rule(&self, d: &T) -> impl Iterator<Item = (&Self, T)> {
        let derivatives = self
            .history
            .last_fn
            .as_ref()
            .map(|f| match f {
                Function::B(b) => {
                    let (da, db) = b.backward(&self.history.ctx, d);
                    vec![da, db]
                }
                Function::U(u) => {
                    let da = u.backward(&self.history.ctx, d);
                    vec![da]
                }
            })
            .unwrap_or_default();
        let inputs = &self.history.inputs;
        // expand derivatives b/c out of bwd is a different size than in of fwd
        inputs
            .iter()
            .zip(derivatives)
            .filter_map(|(i, d)| i.data.expand(d).map(|o| (i, o)))
    }

    // TODO: make iterative
    fn topological_sort_dfs(&self) -> impl Iterator<Item = &Self> {
        let mut q = VecDeque::new();
        let mut visited = HashSet::new();
        fn dfs<'a, E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
            t: &'a Tensor<E, BT, T>,
            visited: &mut HashSet<&'a str>,
            q: &mut VecDeque<&'a Tensor<E, BT, T>>,
        ) {
            if visited.contains(&t.id.as_str()) {
                return;
            }
            visited.insert(&t.id);
            for parent in t.parents() {
                if !parent.is_constant {
                    dfs(parent, visited, q);
                }
            }
            q.push_front(t);
        }
        dfs(self, &mut visited, &mut q);
        q.into_iter()
    }

    fn topological_sort(&self) -> impl Iterator<Item = &Self> {
        let mut queue = VecDeque::new();
        queue.push_back(self);
        let mut visited = HashSet::from([&self.id]);
        let mut result = Vec::new();
        while let Some(var) = queue.pop_front() {
            for parent in var.parents() {
                if !visited.contains(&parent.id) && !parent.is_constant {
                    visited.insert(&parent.id);
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

    pub fn item(&self) -> Option<E> {
        self.data.first()
    }

    fn parents(&self) -> impl Iterator<Item = &Self> {
        self.history.inputs.iter()
    }

    fn is_leaf(&self) -> bool {
        self.history.last_fn.is_none()
    }

    pub fn lt(self, rhs: Tensor<E, BT, T>) -> Self {
        Forward::binary(Lt {}, self, rhs)
    }

    pub fn gt(self, rhs: Tensor<E, BT, T>) -> Self {
        Forward::binary(Lt {}, rhs, self)
    }

    pub fn eq(self, rhs: Tensor<E, BT, T>) -> Self {
        Forward::binary(Eq {}, self, rhs)
    }

    pub fn mm(self, other: Tensor<E, BT, T>) -> Self {
        let mut both_2d = 0;
        let self_shape = self.data.shape().clone();
        let other_shape = other.data.shape().clone();
        let new_self = if self_shape.len() == 2 {
            both_2d += 1;
            self.contiguous()
                .view(&Shape::new(vec![1, self_shape[0], self_shape[1]]))
        } else {
            self
        };
        let new_other = if other_shape.len() == 2 {
            both_2d += 1;
            other
                .contiguous()
                .view(&Shape::new(vec![1, other_shape[0], other_shape[1]]))
        } else {
            other
        };

        let res = Forward::binary(MatMul {}, new_self, new_other);

        if both_2d == 2 {
            let res_shape = res.data.shape().clone();
            res.view(&Shape::new(vec![res_shape[1], res_shape[2]]))
        } else {
            res
        }
    }

    pub fn all(self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => Forward::binary(
                All {},
                self,
                Tensor::from_scalar(UnsafeUsizeConvert::unsafe_from(d)),
            ),
            None => {
                let shape = Shape::scalar(self.size());
                let t = self.view(&shape);
                Forward::binary(All {}, t, Tensor::from_scalar(E::zero()))
            }
        }
    }

    pub fn sum(self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => Forward::binary(
                Sum {},
                self,
                Tensor::from_scalar(UnsafeUsizeConvert::unsafe_from(d)),
            ),
            None => {
                let shape = Shape::scalar(self.size());
                let t = self.contiguous().view(&shape);
                Forward::binary(Sum {}, t, Tensor::from_scalar(E::zero()))
            }
        }
    }

    pub fn mean(self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => {
                let d = UnsafeUsizeConvert::unsafe_from(self.data.shape()[d]);
                let div = Self::from_data(<T as TensorData<E>>::from_scalar(d));
                self.sum(dim) / div
            }
            None => {
                let s = UnsafeUsizeConvert::unsafe_from(self.size());
                let div = Self::from_data(<T as TensorData<E>>::from_scalar(s));
                self.sum(None) / div
            }
        }
    }

    pub fn permute(self, order: Order) -> Self {
        let fs = order
            .data
            .iter()
            .map(|&u| UnsafeUsizeConvert::unsafe_from(u))
            .collect::<Vec<_>>();
        Forward::binary(Permute {}, self, Tensor::from_1d(&fs))
    }

    pub fn view(self, shape: &Shape) -> Self {
        let fs = shape
            .data()
            .iter()
            .map(|&u| UnsafeUsizeConvert::unsafe_from(u))
            .collect::<Vec<_>>();
        Forward::binary(View {}, self, Tensor::from_1d(&fs))
    }

    pub fn contiguous(self) -> Self {
        Forward::unary(Copy {}, self)
    }

    pub fn is_close(self, rhs: Tensor<E, BT, T>) -> Self {
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
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType> Tensor<E, BT, CpuTensorData>
where
    CpuTensorData: Backend<E, BT>,
{
    pub fn arbitrary() -> impl Strategy<Value = Self> {
        CpuTensorData::arbitrary().prop_map(Self::from_data)
    }

    pub fn arbitrary_with_shape(shape: Shape) -> impl Strategy<Value = Self> {
        CpuTensorData::arbitrary_with_shape(shape).prop_map(Self::from_data)
    }

    pub fn arbitrary_tuple() -> impl Strategy<Value = (Self, Self)> {
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
                    Self::from_data(CpuTensorData::new(data1, shape.clone(), strides.clone())),
                    Self::from_data(CpuTensorData::new(data2, shape, strides)),
                )
            })
    }

    pub fn arbitrary_with_order() -> impl Strategy<Value = (Self, Order)> {
        Self::arbitrary().prop_flat_map(|t| {
            let len = t.data.shape.len();
            let ord = collection::vec(0..len, len)
                .prop_shuffle()
                .prop_filter_map("order does not fit", Order::new);
            (Just(t), ord)
        })
    }
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> ops::Add<Tensor<E, BT, T>>
    for Tensor<E, BT, T>
{
    type Output = Tensor<E, BT, T>;

    fn add(self, rhs: Tensor<E, BT, T>) -> Self::Output {
        Forward::binary(Add {}, self, rhs)
    }
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> ops::Sub<Tensor<E, BT, T>>
    for Tensor<E, BT, T>
{
    type Output = Tensor<E, BT, T>;

    fn sub(self, rhs: Tensor<E, BT, T>) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs);
        Forward::binary(Add {}, self, new_rhs)
    }
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> ops::Mul<Tensor<E, BT, T>>
    for Tensor<E, BT, T>
{
    type Output = Tensor<E, BT, T>;

    fn mul(self, rhs: Tensor<E, BT, T>) -> Self::Output {
        Forward::binary(Mul {}, self, rhs)
    }
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> ops::Div<Tensor<E, BT, T>>
    for Tensor<E, BT, T>
{
    type Output = Tensor<E, BT, T>;

    fn div(self, rhs: Tensor<E, BT, T>) -> Self::Output {
        let new_rhs = Forward::unary(Inv {}, rhs);
        Forward::binary(Mul {}, self, new_rhs)
    }
}

impl<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> ops::Neg
    for Tensor<E, BT, T>
{
    type Output = Tensor<E, BT, T>;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend_type::{Par, Seq},
        shaping::idx::Idx,
    };

    use super::*;

    fn unary_grad_assert<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
        F: Fn(Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
    >(
        tensor: Tensor<f64, BT, T>,
        f: F,
    ) {
        let id = &tensor.id.clone();
        let reset = tensor.grad(None).history(History::default());
        let idx = reset.data.shape().sample();
        let out = f(reset);
        let mut res = out.sum(None).backward();
        let tensor_after = res.remove(id);
        assert!(tensor_after.is_some(), "tensor should be in backprop map");
        let unwrapped = tensor_after.unwrap();
        let check = unary_grad_central_diff(unwrapped.clone(), f, &idx);
        assert!(unwrapped.grad.is_some(), "tensor should have a grad");
        let grad = unwrapped.grad.unwrap().data[idx];
        assert!(
            grad.is_close(check),
            "tensor grad ({grad:?}) should be close to central diff ({check:?})",
        );
    }

    fn binary_grad_assert<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
        F: Fn(Tensor<f64, BT, T>, Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
    >(
        tensor1: Tensor<f64, BT, T>,
        tensor2: Tensor<f64, BT, T>,
        f: F,
    ) {
        let (id1, id2) = (&tensor1.id.clone(), &tensor2.id.clone());
        let (reset1, reset2) = (
            tensor1.grad(None).history(History::default()),
            tensor2.grad(None).history(History::default()),
        );
        let (idx1, idx2) = (reset1.data.shape().sample(), reset2.data.shape().sample());
        let out = f(reset1, reset2);
        let mut res = out.sum(None).backward();
        let (after1, after2) = (res.remove(id1), res.remove(id2));
        assert!(
            after1.is_some() && after2.is_some(),
            "tensors should be in backprop map"
        );
        let (unwrapped1, unwrapped2) = (after1.unwrap(), after2.unwrap());
        let (check1, check2) = (
            binary_grad_central_diff(unwrapped1.clone(), unwrapped2.clone(), &f, &idx1, true),
            binary_grad_central_diff(unwrapped1.clone(), unwrapped2.clone(), f, &idx2, false),
        );
        assert!(
            unwrapped1.grad.is_some() && unwrapped2.grad.is_some(),
            "tensors should have grads"
        );
        let (grad1, grad2) = (
            unwrapped1.grad.clone().unwrap().data[idx1],
            unwrapped2.grad.clone().unwrap().data[idx2],
        );
        assert!(
            grad1.is_close(check1),
            "tensor 1 grad ({grad1:?}) should be close to central diff ({check1:?})",
        );
        assert!(
            grad2.is_close(check2),
            "tensor 2 grad ({grad2:?}) should be close to central diff ({check2:?})",
        );
    }

    fn unary_grad_central_diff<
        BT: BackendType,
        T: Backend<f64, BT>,
        F: Fn(Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
    >(
        tensor: Tensor<f64, BT, T>,
        f: F,
        index: &Idx,
    ) -> f64 {
        let eps = 1e-6;
        let shape = tensor.data.shape().clone();
        let up = Tensor::from_data(<T as TensorData<f64>>::epsilon(shape, index, eps));
        let add = tensor.clone() + up.clone();
        let sub = tensor - up;
        let delta = f(add).sum(None) - f(sub).sum(None);

        delta.item().unwrap_or(0.) / (2. * eps)
    }

    fn binary_grad_central_diff<
        BT: BackendType,
        T: Backend<f64, BT>,
        F: Fn(Tensor<f64, BT, T>, Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
    >(
        tensor1: Tensor<f64, BT, T>,
        tensor2: Tensor<f64, BT, T>,
        f: F,
        index: &Idx,
        first: bool,
    ) -> f64 {
        let eps = 1e-6;
        let shape = if first {
            tensor1.data.shape().clone()
        } else {
            tensor2.data.shape().clone()
        };
        let up = Tensor::from_data(<T as TensorData<f64>>::epsilon(shape, index, eps));
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

    fn unary_assert<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
        FT: Fn(Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
        FF: Fn(f64) -> f64,
    >(
        t: Tensor<f64, BT, T>,
        ft: FT,
        ff: FF,
    ) {
        let data = t.data.clone();
        let res = ft(t);
        for idx in res.data.indices() {
            assert!(res.data[idx.clone()].is_close(ff(data[idx])));
        }
    }

    fn binary_assert<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
        FT: Fn(Tensor<f64, BT, T>, Tensor<f64, BT, T>) -> Tensor<f64, BT, T>,
        FF: Fn(f64, f64) -> f64,
    >(
        t1: Tensor<f64, BT, T>,
        t2: Tensor<f64, BT, T>,
        ft: FT,
        ff: FF,
    ) {
        let data1 = t1.data.clone();
        let data2 = t2.data.clone();
        let res = ft(t1, t2);
        for idx in res.data.indices() {
            assert!(res.data[idx.clone()].is_close(ff(data1[idx.clone()], data2[idx])));
        }
    }

    fn permute_grad_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t: Tensor<f64, BT, T>,
        o: Order,
    ) {
        unary_grad_assert(t, move |t| t.permute(o.clone()));
    }

    fn reduce_grad_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t: Tensor<f64, BT, T>,
    ) {
        unary_grad_assert(t.clone(), |t| t.sum(Some(0)));
        unary_grad_assert(t.clone(), |t| t.mean(Some(0)));
        unary_grad_assert(t.clone(), |t| t.mean(None));
    }

    fn binary_grad_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t1: Tensor<f64, BT, T>,
        t2: Tensor<f64, BT, T>,
    ) {
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2);
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2);
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2);
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| {
            t1 / (t2 + Tensor::from_scalar(5.5))
        });
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.gt(t2));
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.lt(t2));
        binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.eq(t2));
    }

    fn binary_grad_broadcast_test<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
    >(
        t1: Tensor<f64, BT, T>,
        t2: Tensor<f64, BT, T>,
    ) {
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 + t2);
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 + t2);
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 - t2);
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 - t2);
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 * t2);
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 * t2);
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| {
            t1 / (t2 + Tensor::from_scalar(5.5))
        });
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| {
            t1 / (t2 + Tensor::from_scalar(5.5))
        });
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.gt(t2));
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.gt(t2));
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.lt(t2));
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.lt(t2));
        binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.eq(t2));
        binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.eq(t2));
    }

    fn unary_grad_complex_test1_<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
    >(
        t: Tensor<f64, BT, T>,
    ) {
        let ft_seq = |t: Tensor<f64, BT, T>| {
            (t.clone() + Tensor::from_scalar(100000.)).ln() + (t - Tensor::from_scalar(200.)).exp()
        };
        unary_grad_assert(t.clone(), ft_seq);
    }

    fn unary_grad_complex_test2_<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
    >(
        t: Tensor<f64, BT, T>,
    ) {
        let ft = |t: Tensor<f64, BT, T>| {
            ((((t * Tensor::from_scalar(10.) + Tensor::from_scalar(7.)).relu()
                * Tensor::from_scalar(6.)
                + Tensor::from_scalar(5.))
            .relu()
                * Tensor::from_scalar(10.))
            .sigmoid())
            .ln()
                / Tensor::from_scalar(50.)
        };
        unary_grad_assert(t.clone(), ft);
    }

    fn unary_grad_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t: Tensor<f64, BT, T>,
    ) {
        unary_grad_assert(t.clone(), |t| -t);
        unary_grad_assert(t.clone(), |t| t.clone() * t);
        unary_grad_assert(t.clone(), |t| t.clone() * t.clone() * t);
        unary_grad_assert(t.clone(), |t| (t + Tensor::from_scalar(3.5)).inv());
        unary_grad_assert(t.clone(), |t| t.sigmoid());
        unary_grad_assert(t.clone(), |t| (t + Tensor::from_scalar(100000.)).ln());
        unary_grad_assert(t.clone(), |t| t.relu());
        unary_grad_assert(t.clone(), |t| t.exp());
    }

    fn unary_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t: Tensor<f64, BT, T>,
    ) {
        unary_assert(t.clone(), |t| -t, |f| -f);
        unary_assert(t.clone(), |t| t.clone() * t, |f| f * f);
        unary_assert(t.clone(), |t| t.clone() * t.clone() * t, |f| f * f * f);
        unary_assert(
            t.clone(),
            |t| t.inv(),
            |f| if f != 0. { 1. / f } else { 0. },
        );
        unary_assert(t.clone(), |t| t.sigmoid(), |f| f.sig());
        unary_assert(t.clone(), |t| t.ln(), |f| if f > 0. { f.ln() } else { 0. });
        unary_assert(t.clone(), |t| t.relu(), |f| f.relu());
        unary_assert(t.clone(), |t| t.exp(), |f| f.exp());
    }

    fn unary_complex_test1_<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
    >(
        t: Tensor<f64, BT, T>,
    ) {
        let ft = |t: Tensor<f64, BT, T>| {
            (t.clone() + Tensor::from_scalar(100000.)).ln() + (t - Tensor::from_scalar(200.)).exp()
        };
        let ff = |f: f64| (f + 100000.).ln() + (f - 200.).exp();
        unary_assert(t.clone(), ft, ff);
    }

    fn unary_complex_test2_<
        BT: BackendType,
        T: Backend<f64, BT> + ops::Index<Idx, Output = f64>,
    >(
        t: Tensor<f64, BT, T>,
    ) {
        let ft = |t: Tensor<f64, BT, T>| {
            ((((t * Tensor::from_scalar(10.) + Tensor::from_scalar(7.)).relu()
                * Tensor::from_scalar(6.)
                + Tensor::from_scalar(5.))
            .relu()
                * Tensor::from_scalar(10.))
            .sigmoid())
            .ln()
                / Tensor::from_scalar(50.)
        };
        let ff = |f: f64| ((((f * 10. + 7.).relu() * 6. + 5.).relu() * 10.).sig()).ln() / 50.;
        unary_assert(t.clone(), ft, ff);
    }

    fn binary_test<BT: BackendType, T: Backend<f64, BT> + ops::Index<Idx, Output = f64>>(
        t1: Tensor<f64, BT, T>,
        t2: Tensor<f64, BT, T>,
    ) {
        binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2, |f1, f2| f1 + f2);
        binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2, |f1, f2| f1 - f2);
        binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2, |f1, f2| f1 * f2);
        binary_assert(
            t1.clone(),
            t2.clone(),
            |t1, t2| t1 / t2,
            |f1, f2| if f2 == 0. { 0. } else { f1 / f2 },
        );
        binary_assert(
            t1.clone(),
            t2.clone(),
            |t1, t2| t1.gt(t2),
            |f1, f2| if f2 < f1 { 1. } else { 0. },
        );
        binary_assert(
            t1.clone(),
            t2.clone(),
            |t1, t2| t1.lt(t2),
            |f1, f2| if f1 < f2 { 1. } else { 0. },
        );
        binary_assert(
            t1.clone(),
            t2.clone(),
            |t1, t2| t1.eq(t2),
            |f1, f2| if f1 == f2 { 1. } else { 0. },
        );
    }

    proptest! {
        #[test]
        fn matmul_tests(
            a_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary_with_shape(Shape::new(vec![2, 3])),
            b_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary_with_shape(Shape::new(vec![3, 4])),
            a_par in Tensor::<f64, Par, CpuTensorData>::arbitrary_with_shape(Shape::new(vec![2, 3])),
            b_par in Tensor::<f64, Par, CpuTensorData>::arbitrary_with_shape(Shape::new(vec![3, 4])),
        ) {
            let c_seq = a_seq.clone().mm(b_seq.clone());
            let cprime_seq = (
                a_seq.clone().view(&Shape::new(vec![2, 3, 1])) *
                b_seq.clone().view(&Shape::new(vec![1, 3, 4]))
            ).sum(Some(1)).view(&Shape::new(vec![2, 4]));
            for idx in c_seq.data.indices() {
                assert!(c_seq.data[idx.clone()].is_close(cprime_seq.data[idx]));
            }
            binary_grad_assert(a_seq.clone(), b_seq.clone(), |t1, t2| t1.mm(t2));

            let c_par = a_par.clone().mm(b_par.clone());
            let cprime_par = (
                a_par.clone().view(&Shape::new(vec![2, 3, 1])) *
                b_par.clone().view(&Shape::new(vec![1, 3, 4]))
            ).sum(Some(1)).view(&Shape::new(vec![2, 4]));
            for idx in c_par.data.indices() {
                assert!(c_par.data[idx.clone()].is_close(cprime_par.data[idx]));
            }
            binary_grad_assert(a_par.clone(), b_par.clone(), |t1, t2| t1.mm(t2));
        }

        // TODO: reimplement backward
        // #[test]
        fn permute_grad_tests(
            (t_seq, o_seq) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_with_order(),
            (t_par, o_par) in Tensor::<f64, Par, CpuTensorData>::arbitrary_with_order(),
        ) {
            permute_grad_test(t_seq, o_seq);
            permute_grad_test(t_par, o_par);
        }

        #[test]
        fn reduce_grad_tests(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            reduce_grad_test(t_seq);
            reduce_grad_test(t_par);
        }

        #[test]
        fn binary_grad_tests(
            (t1_seq, t2_seq) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_tuple(),
            (t1_par, t2_par) in Tensor::<f64, Par, CpuTensorData>::arbitrary_tuple(),
        ) {
            binary_grad_test(t1_seq, t2_seq);
            binary_grad_test(t1_par, t2_par);
        }

        #[test]
        fn binary_grad_broadcast_tests(
            (t1_seq, t2_seq) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_tuple(),
            (t1_par, t2_par) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_tuple(),
        ) {
            binary_grad_test(t1_seq, t2_seq);
            binary_grad_test(t1_par, t2_par);
        }

        #[test]
        fn unary_grad_complex_test1(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            unary_grad_complex_test1_(t_seq);
            unary_grad_complex_test1_(t_par);
        }

        #[test]
        fn unary_grad_complex_test2(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            unary_grad_complex_test2_(t_seq);
            unary_grad_complex_test2_(t_par);
        }

        #[test]
        fn unary_grad_tests(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
        ) {
            unary_grad_test(t_seq);
            unary_grad_test(t_par);
        }

        #[test]
        fn binary_tests(
            (t1_seq, t2_seq) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_tuple(),
            (t1_par, t2_par) in Tensor::<f64, Seq, CpuTensorData>::arbitrary_tuple(),
        ) {
            binary_test(t1_seq, t2_seq);
            binary_test(t1_par, t2_par);
        }

        #[test]
        fn unary_complex_test1(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            unary_complex_test1_(t_seq);
            unary_complex_test1_(t_par);
        }

        #[test]
        fn unary_complex_test2(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            unary_complex_test2_(t_seq);
            unary_complex_test2_(t_par);
        }

        #[test]
        fn unary_tests(
            t_seq in Tensor::<f64, Seq, CpuTensorData>::arbitrary(),
            t_par in Tensor::<f64, Par, CpuTensorData>::arbitrary(),
        ) {
            unary_test(t_seq);
            unary_test(t_par);
        }
    }

    fn view_test<BT: BackendType, T: Backend<f64, BT>>(t: Tensor<f64, BT, T>) {
        assert_eq!(&Shape::new(vec![2, 3]), t.data.shape());
        let t2 = t.clone().view(&Shape::new(vec![6]));
        assert_eq!(&Shape::new(vec![6]), t2.data.shape());
        let t3 = t2.view(&Shape::new(vec![1, 6]));
        assert_eq!(&Shape::new(vec![1, 6]), t3.data.shape());
        let t4 = t3.view(&Shape::new(vec![6, 1]));
        assert_eq!(&Shape::new(vec![6, 1]), t4.data.shape());
        let t5 = t4.view(&Shape::new(vec![2, 3]));
        assert_eq!(&Shape::new(vec![2, 3]), t5.data.shape());
        assert_eq!(Some(1.), t.is_close(t5).all(None).item());
    }

    #[test]
    fn test_view() {
        let t_seq =
            Tensor::<f64, Seq, CpuTensorData>::from_2d(&[&[2., 3., 4.], &[4., 5., 7.]]).unwrap();
        view_test(t_seq);
        let t_par =
            Tensor::<f64, Par, CpuTensorData>::from_2d(&[&[2., 3., 4.], &[4., 5., 7.]]).unwrap();
        view_test(t_par);
    }

    fn reduce_forward_one_dim_test<BT: BackendType, T: Backend<f64, BT>>(t: Tensor<f64, BT, T>) {
        let summed = t.sum(Some(0));
        let exp = Tensor::from_1d(&[11., 16.]);
        let is_close = summed.is_close(exp);
        let shape = Shape::scalar(is_close.size());
        assert_eq!(Some(1.), is_close.view(&shape).all(Some(0)).item());
    }

    #[test]
    fn test_reduce_forward_one_dim() {
        let shape = Shape::new(vec![3, 2]);
        let strides = (&shape).into();
        let td = CpuTensorData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);

        let t_seq: Tensor<f64, Seq, _> = Tensor::from_data(td.clone());
        reduce_forward_one_dim_test(t_seq);
        let t_par: Tensor<f64, Par, _> = Tensor::from_data(td.clone());
        reduce_forward_one_dim_test(t_par);
    }

    fn reduce_forward_one_dim_2_test<BT: BackendType, T: Backend<f64, BT>>(t: Tensor<f64, BT, T>) {
        let summed = t.sum(Some(1));
        let exp = Tensor::from_data(TensorData::from_2d(&[&[5.], &[10.], &[12.]]).unwrap());
        let is_close = summed.is_close(exp);
        let shape = Shape::new(vec![is_close.size()]);
        assert_eq!(Some(1.), is_close.view(&shape).all(Some(0)).item());
    }

    #[test]
    fn test_reduce_forward_one_dim_2() {
        let shape = Shape::new(vec![3, 2]);
        let strides = (&shape).into();
        let td = CpuTensorData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);

        let t_seq: Tensor<f64, Seq, _> = Tensor::from_data(td.clone());
        reduce_forward_one_dim_2_test(t_seq);
        let t_par: Tensor<f64, Par, _> = Tensor::from_data(td);
        reduce_forward_one_dim_2_test(t_par);
    }

    fn reduce_forward_all_dim_test<BT: BackendType, T: Backend<f64, BT>>(t: Tensor<f64, BT, T>) {
        let summed = t.sum(None);
        assert_eq!(Some(27.), summed.item());
    }

    #[test]
    fn test_reduce_forward_all_dim() {
        let shape = Shape::new(vec![3, 2]);
        let t_seq = Tensor::<f64, Seq, CpuTensorData>::from_1d(&[2., 3., 4., 6., 5., 7.])
            .reshape(shape.clone());
        reduce_forward_all_dim_test(t_seq);
        let t_par =
            Tensor::<f64, Par, CpuTensorData>::from_1d(&[2., 3., 4., 6., 5., 7.]).reshape(shape);
        reduce_forward_all_dim_test(t_par);
    }
}
