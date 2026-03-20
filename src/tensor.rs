use crate::{
    autodiff::{forward::Forward, gradients::Gradients, trace::Trace},
    backend::{
        backend::{Backend, GpuBackend},
        mode::Mode,
    },
    fns::{
        binary::{Add, All, Eq, IsClose, Lt, MatMul, Mul, Permute, Sum, View},
        function::Function,
        unary::{Copy, Exp, Inv, Ln, Neg, Relu, Sig},
    },
    math::element::Element,
    ops::ops::Ops,
    shaping::{order::Order, shape::Shape, strides::Strides},
    storage::{cpu_data::CpuData, data::Data, gpu_data::GpuData},
    util::unsafe_usize_convert::UnsafeUsizeConvert,
    wgpu::wgpu_context::get_wgpu_context,
};
use proptest::{collection, prelude::*};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    ops,
    sync::atomic::{AtomicU64, Ordering},
};

static TENSOR_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct Tensor<'a, B>
where
    B: Backend + 'a,
{
    pub data: B::Storage<'a>,
    pub grad: Option<Box<Tensor<'a, B>>>,
    pub trace: Trace<'a, B>,
    pub id: u64,
    pub is_constant: bool,
    _marker: PhantomData<(B::Element, B::Mode)>,
}

impl<'a, B: Backend> Tensor<'a, B> {
    pub fn new(data: B::Storage<'a>, trace: Trace<'a, B>) -> Self {
        let id = TENSOR_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            data,
            grad: None,
            trace,
            id,
            is_constant: false,
            _marker: PhantomData,
        }
    }

    pub fn from_data(data: B::Storage<'a>) -> Self {
        let id = TENSOR_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            data,
            grad: None,
            trace: Trace::default(),
            id,
            is_constant: false,
            _marker: PhantomData,
        }
    }

    pub fn ones(shape: Shape) -> Self {
        Self::from_data(<B::Storage<'a> as Data<B::Element>>::ones(shape))
    }

    pub fn from_scalar(data: B::Element) -> Self {
        Self::from_data(<B::Storage<'a> as Data<B::Element>>::from_scalar(data)).make_constant()
    }

    pub fn from_shape(data: &[B::Element], shape: Shape) -> Self {
        Self::from_data(<B::Storage<'a> as Data<B::Element>>::from_shape(
            data, shape,
        ))
    }

    pub fn from_1d(data: &[B::Element]) -> Self {
        Self::from_data(<B::Storage<'a> as Data<B::Element>>::from_1d(data))
    }

    pub fn from_2d(data: &[&[B::Element]]) -> Option<Self> {
        <B::Storage<'a> as Data<B::Element>>::from_2d(data).map(Self::from_data)
    }

    pub fn from_tuples(data: &[(B::Element, B::Element)]) -> Self {
        let len = data.len();
        let d = data.iter().fold(Vec::with_capacity(len * 2), |mut acc, t| {
            acc.push(t.0);
            acc.push(t.1);
            acc
        });
        let shape = Shape::new(vec![len, 2]);
        let strides = (&shape).into();
        Self::from_data(<B::Storage<'a> as Data<B::Element>>::from(
            &d, shape, strides,
        ))
    }

    pub fn trace(mut self, t: Trace<'a, B>) -> Self {
        self.trace = t;
        self
    }

    pub fn grad(mut self, grad: Option<Tensor<'a, B>>) -> Self {
        self.grad = grad.map(Box::new);
        self
    }

    pub fn data(mut self, data: B::Storage<'a>) -> Self {
        self.data = data;
        self
    }

    pub fn id(mut self, id: u64) -> Self {
        self.id = id;
        self
    }

    fn make_constant(mut self) -> Self {
        self.is_constant = true;
        self
    }

    pub fn backward(&self) -> Gradients<'a, B> {
        assert!(
            *self.data.shape() == Shape::new(vec![1]),
            "use backprop for non-scalar tensors"
        );
        self.backprop(Self::from_scalar(B::Element::one()))
    }

    pub fn backprop(&self, d: Tensor<'a, B>) -> Gradients<'a, B> {
        let sorted = self.topological_sort_dfs();
        let mut derivs = HashMap::from([(&self.id, d)]);
        let mut res: HashMap<u64, Self> = HashMap::new();
        for s in sorted {
            if let Some(current_deriv) = derivs.get(&s.id).cloned() {
                for (parent, grad) in s.chain_rule(&current_deriv.data) {
                    let grad_tensor = Tensor::from_data(grad).make_constant();
                    if parent.is_leaf() {
                        let new = match res.get(&parent.id) {
                            Some(s) => s.clone().accumulate_derivative(grad_tensor),
                            None => parent.clone().accumulate_derivative(grad_tensor),
                        };
                        res.insert(parent.id, new);
                    } else {
                        match derivs.remove(&parent.id) {
                            Some(e) => derivs.insert(&parent.id, e + grad_tensor),
                            None => derivs.insert(&parent.id, grad_tensor),
                        };
                    }
                }
            }
        }
        Gradients(res)
    }

    fn accumulate_derivative(mut self, d: Tensor<'a, B>) -> Self {
        if self.is_leaf() {
            let grad = self.grad.map(|t| *t + d.clone()).unwrap_or(d);
            self.grad = Some(Box::new(grad.clone()));
            self
        } else {
            self
        }
    }

    fn chain_rule(&self, d: &B::Storage<'a>) -> impl Iterator<Item = (&Self, B::Storage<'a>)> {
        let inputs = &self.trace.inputs;
        let gradients = self
            .trace
            .last_fn
            .as_ref()
            .map(|f| match f {
                Function::B(b) => {
                    let (da, db) = b.backward(&inputs[0].data, &inputs[1].data, d);
                    vec![da, db]
                }
                Function::U(u) => {
                    let da = u.backward(&inputs[0].data, d);
                    vec![da]
                }
            })
            .unwrap_or_default();
        // expand derivatives b/c out of bwd is a different size than in of fwd
        inputs
            .iter()
            .zip(gradients)
            .filter_map(|(i, d)| i.data.expand(d).map(|o| (i, o)))
    }

    // TODO: make iterative
    fn topological_sort_dfs(&self) -> impl Iterator<Item = &Self> {
        let mut q = VecDeque::new();
        let mut visited = HashSet::new();
        fn dfs<'a, 'b, B: Backend>(
            t: &'b Tensor<'a, B>,
            visited: &mut HashSet<u64>,
            q: &mut VecDeque<&'b Tensor<'a, B>>,
        ) {
            if visited.contains(&t.id) {
                return;
            }
            visited.insert(t.id);
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

    //fn topological_sort(&self) -> impl Iterator<Item = &Self> {
    //    let mut queue = VecDeque::new();
    //    queue.push_back(self);
    //    let mut visited = HashSet::from([&self.id]);
    //    let mut result = Vec::new();
    //    while let Some(var) = queue.pop_front() {
    //        for parent in var.parents() {
    //            if !visited.contains(&parent.id) && !parent.is_constant {
    //                visited.insert(&parent.id);
    //                queue.push_back(parent);
    //            }
    //        }
    //        result.push(var);
    //    }
    //    result.into_iter()
    //}

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.data = self.data.reshape(shape);
        self
    }

    pub fn size(&self) -> usize {
        self.data.size()
    }

    pub fn item(&self) -> Option<B::Element> {
        self.data.first()
    }

    fn parents(&self) -> impl Iterator<Item = &Self> {
        self.trace.inputs.iter()
    }

    fn is_leaf(&self) -> bool {
        self.trace.last_fn.is_none()
    }

    pub fn lt(self, rhs: Tensor<'a, B>) -> Self {
        Forward::binary(Lt {}, self, rhs)
    }

    pub fn gt(self, rhs: Tensor<'a, B>) -> Self {
        Forward::binary(Lt {}, rhs, self)
    }

    pub fn eq(self, rhs: Tensor<'a, B>) -> Self {
        Forward::binary(Eq {}, self, rhs)
    }

    pub fn mm(self, other: Tensor<'a, B>) -> Self {
        let self_shape = self.data.shape().clone();
        let other_shape = other.data.shape().clone();
        let both_2d = self_shape.len() == 2 && other_shape.len() == 2;

        let new_self = if self_shape.len() == 2 {
            self.contiguous()
                .view(&Shape::new(vec![1, self_shape[0], self_shape[1]]))
        } else {
            self
        };
        let new_other = if other_shape.len() == 2 {
            other
                .contiguous()
                .view(&Shape::new(vec![1, other_shape[0], other_shape[1]]))
        } else {
            other
        };

        let res = Forward::binary(MatMul {}, new_self, new_other);

        if both_2d {
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
                Forward::binary(All {}, t, Tensor::from_scalar(B::Element::zero()))
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
                Forward::binary(Sum {}, t, Tensor::from_scalar(B::Element::zero()))
            }
        }
    }

    pub fn mean(self, dim: Option<usize>) -> Self {
        match dim {
            Some(d) => {
                let d = UnsafeUsizeConvert::unsafe_from(self.data.shape()[d]);
                let div = Self::from_data(<B::Storage<'a> as Data<B::Element>>::from_scalar(d));
                self.sum(dim) / div
            }
            None => {
                let s = UnsafeUsizeConvert::unsafe_from(self.size());
                let div = Self::from_data(<B::Storage<'a> as Data<B::Element>>::from_scalar(s));
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

    pub fn is_close(self, rhs: Tensor<'a, B>) -> Self {
        Forward::binary(IsClose {}, self, rhs)
    }

    pub fn sig(self) -> Self {
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

// TODO: find a way to make this drier
impl<'a, B> Tensor<'a, B>
where
    B: Backend<Storage<'a> = CpuData>,
{
    pub fn arbitrary() -> impl Strategy<Value = Self> {
        CpuData::arbitrary().prop_map(Self::from_data)
    }

    pub fn arbitrary_no_zero() -> impl Strategy<Value = Self> {
        CpuData::arbitrary_no_zero().prop_map(Self::from_data)
    }

    pub fn arbitrary_matmul_tuple() -> impl Strategy<Value = (Self, Self)> {
        let dim_s = 2_usize..=4;
        (dim_s.clone(), dim_s.clone(), dim_s.clone(), dim_s).prop_flat_map(|(a, b, c, d)| {
            let shape1 = Shape::new(vec![d, a, b]);
            let shape2 = Shape::new(vec![1, b, c]);
            (
                Self::arbitrary_with_shape(shape1),
                Self::arbitrary_with_shape(shape2),
            )
        })
    }

    pub fn arbitrary_with_shape(shape: Shape) -> impl Strategy<Value = Self> {
        CpuData::arbitrary_with_shape(shape).prop_map(Self::from_data)
    }

    pub fn arbitrary_tuple() -> impl Strategy<Value = (Self, Self)> {
        let strategy = -1.0f64..1.;
        Self::arbitrary_tuple_with_strategy(strategy.clone(), strategy)
    }

    // useful when central diff doesn't work lt and gt if x = y
    pub fn arbitrary_disjoint_tuple() -> impl Strategy<Value = (Self, Self)> {
        let s1 = -1.0f64..0.;
        let s2 = 0.0f64..1.;
        Self::arbitrary_tuple_with_strategy(s1, s2)
    }

    fn arbitrary_tuple_with_strategy<S: Strategy<Value = f64> + Clone>(
        s1: S,
        s2: S,
    ) -> impl Strategy<Value = (Self, Self)> {
        Shape::arbitrary()
            .prop_flat_map(move |shape| {
                let size = shape.size;
                let data1 = collection::vec(s1.clone(), size);
                let data2 = collection::vec(s2.clone(), size);
                (data1, data2, Just(shape))
            })
            .prop_map(|(data1, data2, shape)| {
                let strides: Strides = (&shape).into();
                (
                    Self::from_data(CpuData::new(data1, shape.clone(), strides.clone())),
                    Self::from_data(CpuData::new(data2, shape, strides)),
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

impl<'a> Tensor<'a, GpuBackend> {
    pub fn arbitrary() -> impl Strategy<Value = Self> {
        GpuData::arbitrary().prop_map(Self::from_data)
    }

    pub fn arbitrary_no_zero() -> impl Strategy<Value = Self> {
        GpuData::arbitrary_no_zero().prop_map(Self::from_data)
    }

    pub fn arbitrary_matmul_tuple() -> impl Strategy<Value = (Self, Self)> {
        let dim_s = 2_usize..4;
        (dim_s.clone(), dim_s.clone(), dim_s.clone(), dim_s).prop_flat_map(|(a, b, c, d)| {
            let shape1 = Shape::new(vec![d, a, b]);
            let shape2 = Shape::new(vec![1, b, c]);
            (
                Self::arbitrary_with_shape(shape1),
                Self::arbitrary_with_shape(shape2),
            )
        })
    }

    pub fn arbitrary_with_shape(shape: Shape) -> impl Strategy<Value = Self> {
        GpuData::arbitrary_with_shape(shape).prop_map(Self::from_data)
    }

    pub fn arbitrary_tuple() -> impl Strategy<Value = (Self, Self)> {
        let strategy = -1.0f32..1.;
        Self::arbitrary_tuple_with_strategy(strategy.clone(), strategy.clone())
    }

    // useful when central diff doesn't work lt and gt if x = y
    pub fn arbitrary_disjoint_tuple() -> impl Strategy<Value = (Self, Self)> {
        let s1 = -1.0f32..0.;
        let s2 = 0.0f32..1.;
        Self::arbitrary_tuple_with_strategy(s1, s2)
    }

    fn arbitrary_tuple_with_strategy<S: Strategy<Value = f32> + Clone>(
        s1: S,
        s2: S,
    ) -> impl Strategy<Value = (Self, Self)> {
        Shape::arbitrary()
            .prop_flat_map(move |shape| {
                let size = shape.size;
                let data1 = collection::vec(s1.clone(), size);
                let data2 = collection::vec(s2.clone(), size);
                (data1, data2, Just(shape))
            })
            .prop_map(|(data1, data2, shape)| {
                let strides: Strides = (&shape).into();
                (
                    Self::from_data(GpuData::new(
                        &data1,
                        shape.clone(),
                        strides.clone(),
                        get_wgpu_context(),
                    )),
                    Self::from_data(GpuData::new(&data2, shape, strides, get_wgpu_context())),
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

impl<'a, B: Backend> ops::Add<Tensor<'a, B>> for Tensor<'a, B> {
    type Output = Tensor<'a, B>;

    fn add(self, rhs: Tensor<'a, B>) -> Self::Output {
        Forward::binary(Add {}, self, rhs)
    }
}

impl<'a, B: Backend> ops::Sub<Tensor<'a, B>> for Tensor<'a, B> {
    type Output = Tensor<'a, B>;

    fn sub(self, rhs: Tensor<'a, B>) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs);
        Forward::binary(Add {}, self, new_rhs)
    }
}

impl<'a, B: Backend> ops::Mul<Tensor<'a, B>> for Tensor<'a, B> {
    type Output = Tensor<'a, B>;

    fn mul(self, rhs: Tensor<'a, B>) -> Self::Output {
        Forward::binary(Mul {}, self, rhs)
    }
}

impl<'a, B: Backend> ops::Div<Tensor<'a, B>> for Tensor<'a, B> {
    type Output = Tensor<'a, B>;

    fn div(self, rhs: Tensor<'a, B>) -> Self::Output {
        let new_rhs = Forward::unary(Inv {}, rhs);
        Forward::binary(Mul {}, self, new_rhs)
    }
}

impl<'a, B: Backend> ops::Neg for Tensor<'a, B> {
    type Output = Tensor<'a, B>;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self)
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use crate::{
        backend::backend::{CpuParBackend, CpuSeqBackend},
        shaping::idx::Idx,
        storage::gpu_data::GpuData,
    };

    use super::*;

    fn unary_grad_central_diff<'a, B, F>(
        tensor: Tensor<'a, B>,
        f: F,
        index: &Idx,
        eps: B::Element,
    ) -> B::Element
    where
        B: Backend,
        F: Fn(Tensor<'a, B>) -> Tensor<'a, B>,
    {
        let shape = tensor.data.shape().clone();
        let up = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::epsilon(
            shape, index, eps,
        ));
        let add = tensor.clone() + up.clone();
        let sub = tensor - up;
        let delta = f(add).sum(None) - f(sub).sum(None);

        delta.item().unwrap_or(B::Element::zero()) / (B::Element::fromf(2.) * eps)
    }

    fn binary_grad_central_diff<'a, B, F>(
        tensor1: Tensor<'a, B>,
        tensor2: Tensor<'a, B>,
        f: F,
        index: &Idx,
        first: bool,
        eps: B::Element,
    ) -> B::Element
    where
        B: Backend,
        F: Fn(Tensor<'a, B>, Tensor<'a, B>) -> Tensor<'a, B>,
    {
        let shape = if first {
            tensor1.data.shape().clone()
        } else {
            tensor2.data.shape().clone()
        };
        let up = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::epsilon(
            shape, index, eps,
        ));
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

        delta.item().unwrap_or(B::Element::zero()) / (B::Element::fromf(2.) * eps)
    }

    mod cpu {
        use crate::backend::backend::{CpuParBackend, CpuSeqBackend};

        use super::*;

        fn unary_grad_assert<'a, B, F>(tensor: Tensor<'a, B>, f: F)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
            F: Fn(Tensor<'a, B>) -> Tensor<'a, B>,
        {
            let reset = tensor.grad(None).trace(Trace::default());
            let idx = reset.data.shape().sample_idx();
            let out = f(reset.clone());
            let grads = out.sum(None).backward();
            let grad = grads.wrt(&reset).unwrap();
            let grad_data = grad.data[idx.clone()];
            let check = unary_grad_central_diff(reset.clone(), f, &idx, B::Element::fromf(1e-6));
            assert!(
                grad_data.is_close(check),
                "tensor grad ({grad:?}) should be close to central diff ({check:?})",
            );
        }

        pub fn binary_grad_assert<'a, B, F>(tensor1: Tensor<'a, B>, tensor2: Tensor<'a, B>, f: F)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
            F: Fn(Tensor<'a, B>, Tensor<'a, B>) -> Tensor<'a, B>,
        {
            let (reset1, reset2) = (
                tensor1.grad(None).trace(Trace::default()),
                tensor2.grad(None).trace(Trace::default()),
            );
            let (idx1, idx2) = (
                reset1.data.shape().sample_idx(),
                reset2.data.shape().sample_idx(),
            );
            let out = f(reset1.clone(), reset2.clone());
            let grads = out.sum(None).backward();
            let (grad1, grad2) = (grads.wrt(&reset1).unwrap(), grads.wrt(&reset2).unwrap());
            let (grad_data1, grad_data2) = (grad1.data[idx1.clone()], grad2.data[idx2.clone()]);
            let (check1, check2) = (
                binary_grad_central_diff(
                    reset1.clone(),
                    reset2.clone(),
                    &f,
                    &idx1,
                    true,
                    B::Element::fromf(1e-6),
                ),
                binary_grad_central_diff(
                    reset1.clone(),
                    reset2.clone(),
                    f,
                    &idx2,
                    false,
                    B::Element::fromf(1e-6),
                ),
            );
            assert!(
                grad_data1.is_close(check1),
                "tensor 1 grad ({grad1:?}) should be close to central diff ({check1:?})",
            );
            assert!(
                grad_data2.is_close(check2),
                "tensor 2 grad ({grad2:?}) should be close to central diff ({check2:?})",
            );
        }

        fn unary_assert<'a, B, FT, FF>(t: Tensor<'a, B>, ft: FT, ff: FF)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
            FT: Fn(Tensor<'a, B>) -> Tensor<'a, B>,
            FF: Fn(B::Element) -> B::Element,
        {
            let data = t.data.clone();
            let res = ft(t);
            for idx in res.data.indices() {
                assert!(res.data[idx.clone()].is_close(ff(data[idx])));
            }
        }

        fn binary_assert<'a, B, FT, FF>(t1: Tensor<'a, B>, t2: Tensor<'a, B>, ft: FT, ff: FF)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
            FT: Fn(Tensor<'a, B>, Tensor<'a, B>) -> Tensor<'a, B>,
            FF: Fn(B::Element, B::Element) -> B::Element,
        {
            let data1 = t1.data.clone();
            let data2 = t2.data.clone();
            let res = ft(t1, t2);
            for idx in res.data.indices() {
                assert!(res.data[idx.clone()].is_close(ff(data1[idx.clone()], data2[idx])));
            }
        }

        fn permute_grad_test<'a, B>(t: Tensor<'a, B>, o: Order)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            unary_grad_assert(t, move |t| t.permute(o.clone()));
        }

        fn reduce_grad_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            unary_grad_assert(t.clone(), |t| t.sum(Some(0)));
            unary_grad_assert(t.clone(), |t| t.mean(Some(0)));
            unary_grad_assert(t.clone(), |t| t.mean(None));
        }

        fn binary_grad_test<'a, B>(t1: Tensor<'a, B>, t2: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2);
            binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2);
            binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2);
            binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| {
                t1 / (t2 + Tensor::from_scalar(B::Element::fromf(5.5)))
            });
            binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.eq(t2));
        }

        fn binary_grad_broadcast_test<'a, B>(t1: Tensor<'a, B>, t2: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 + t2);
            binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 + t2);
            binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 - t2);
            binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 - t2);
            binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1 * t2);
            binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1 * t2);
            binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| {
                t1 / (t2 + Tensor::from_scalar(B::Element::fromf(5.5)))
            });
            binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| {
                t1 / (t2 + Tensor::from_scalar(B::Element::fromf(5.5)))
            });
            binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.eq(t2));
            binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.eq(t2));
        }

        fn unary_grad_complex_test1_<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            let ft_seq = |t: Tensor<'a, B>| {
                (t.clone() + Tensor::from_scalar(B::Element::fromf(100000.))).ln()
                    + (t - Tensor::from_scalar(B::Element::fromf(200.))).exp()
            };
            unary_grad_assert(t.clone(), ft_seq);
        }

        fn unary_grad_complex_test2_<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            let ft = |t: Tensor<'a, B>| {
                ((((t * Tensor::from_scalar(B::Element::fromf(10.))
                    + Tensor::from_scalar(B::Element::fromf(7.)))
                .relu()
                    * Tensor::from_scalar(B::Element::fromf(6.))
                    + Tensor::from_scalar(B::Element::fromf(5.)))
                .relu()
                    * Tensor::from_scalar(B::Element::fromf(10.)))
                .sig())
                .ln()
                    / Tensor::from_scalar(B::Element::fromf(50.))
            };
            unary_grad_assert(t.clone(), ft);
        }

        fn unary_grad_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            unary_grad_assert(t.clone(), |t| -t);
            unary_grad_assert(t.clone(), |t| t.clone() * t);
            unary_grad_assert(t.clone(), |t| t.clone() * t.clone() * t);
            unary_grad_assert(t.clone(), |t| {
                (t + Tensor::from_scalar(B::Element::fromf(3.5))).inv()
            });
            unary_grad_assert(t.clone(), |t| t.sig());
            unary_grad_assert(t.clone(), |t| {
                (t + Tensor::from_scalar(B::Element::fromf(100000.))).ln()
            });
            unary_grad_assert(t.clone(), |t| t.exp());
        }

        fn unary_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            unary_assert(t.clone(), |t| -t, |f| -f);
            unary_assert(t.clone(), |t| t.clone() * t, |f| f * f);
            unary_assert(t.clone(), |t| t.clone() * t.clone() * t, |f| f * f * f);
            unary_assert(
                t.clone(),
                |t| t.inv(),
                |f| {
                    if f != B::Element::zero() {
                        B::Element::one() / f
                    } else {
                        B::Element::zero()
                    }
                },
            );
            unary_assert(t.clone(), |t| t.sig(), |f| f.sig());
            unary_assert(
                t.clone(),
                |t| t.ln(),
                |f| {
                    if f > B::Element::zero() {
                        f.ln()
                    } else {
                        B::Element::zero()
                    }
                },
            );
            unary_assert(t.clone(), |t| t.relu(), |f| f.relu());
            unary_assert(t.clone(), |t| t.exp(), |f| f.exp());
            unary_assert(t.clone(), |t| t.contiguous(), |f| f);
        }

        fn unary_complex_test1_<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            let ft = |t: Tensor<'a, B>| {
                (t.clone() + Tensor::from_scalar(B::Element::fromf(100000.))).ln()
                    + (t - Tensor::from_scalar(B::Element::fromf(200.))).exp()
            };
            let ff = |f: B::Element| {
                (f + B::Element::fromf(100000.)).ln() + (f - B::Element::fromf(200.)).exp()
            };
            unary_assert(t.clone(), ft, ff);
        }

        fn unary_complex_test2_<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            let ft = |t: Tensor<'a, B>| {
                ((((t * Tensor::from_scalar(B::Element::fromf(10.))
                    + Tensor::from_scalar(B::Element::fromf(7.)))
                .relu()
                    * Tensor::from_scalar(B::Element::fromf(6.))
                    + Tensor::from_scalar(B::Element::fromf(5.)))
                .relu()
                    * Tensor::from_scalar(B::Element::fromf(10.)))
                .sig())
                .ln()
                    / Tensor::from_scalar(B::Element::fromf(50.))
            };
            let ff = |f: B::Element| {
                ((((f * B::Element::fromf(10.) + B::Element::fromf(7.)).relu()
                    * B::Element::fromf(6.)
                    + B::Element::fromf(5.))
                .relu()
                    * B::Element::fromf(10.))
                .sig())
                .ln()
                    / B::Element::fromf(50.)
            };
            unary_assert(t.clone(), ft, ff);
        }

        fn binary_test<'a, B>(t1: Tensor<'a, B>, t2: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2, |f1, f2| f1 + f2);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2, |f1, f2| f1 - f2);
            binary_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2, |f1, f2| f1 * f2);
            binary_assert(
                t1.clone(),
                t2.clone(),
                |t1, t2| t1 / t2,
                |f1, f2| {
                    if f2 == B::Element::zero() {
                        B::Element::zero()
                    } else {
                        f1 / f2
                    }
                },
            );
            binary_assert(
                t1.clone(),
                t2.clone(),
                |t1, t2| t1.gt(t2),
                |f1, f2| {
                    if f2 < f1 {
                        B::Element::one()
                    } else {
                        B::Element::zero()
                    }
                },
            );
            binary_assert(
                t1.clone(),
                t2.clone(),
                |t1, t2| t1.lt(t2),
                |f1, f2| {
                    if f1 < f2 {
                        B::Element::one()
                    } else {
                        B::Element::zero()
                    }
                },
            );
            binary_assert(
                t1.clone(),
                t2.clone(),
                |t1, t2| t1.eq(t2),
                |f1, f2| {
                    if f1 == f2 {
                        B::Element::one()
                    } else {
                        B::Element::zero()
                    }
                },
            );
        }

        fn matmul_test<'a, B>(t1: Tensor<'a, B>, t2: Tensor<'a, B>)
        where
            B: Backend,
            B::Storage<'a>: ops::Index<Idx, Output = B::Element>,
        {
            if let [d, a, b] = t1.data.shape().data()
                && let [_, _, c] = t2.data.shape().data()
            {
                let c_mm = t1.clone().mm(t2.clone());
                let c_prime = (t1
                    .clone()
                    .contiguous()
                    .view(&Shape::new(vec![*d, *a, *b, 1]))
                    * t2.clone()
                        .contiguous()
                        .view(&Shape::new(vec![1, 1, *b, *c])))
                .sum(Some(2))
                .view(&Shape::new(vec![*d, *a, *c]));
                assert_eq!(
                    Some(B::Element::one()),
                    c_mm.is_close(c_prime).all(None).item()
                );
            } else {
                panic!("both tensors should be 3d");
            }
        }

        proptest! {
            #[test]
            fn matmul_tests(
                (a_seq, b_seq) in Tensor::<CpuSeqBackend>::arbitrary_matmul_tuple(),
                (a_par, b_par) in Tensor::<CpuParBackend>::arbitrary_matmul_tuple(),
            ) {
                matmul_test(a_seq, b_seq);
                matmul_test(a_par, b_par);
            }

            #[test]
            fn matmul_grad_tests(
                (a_seq, b_seq) in Tensor::<CpuSeqBackend>::arbitrary_matmul_tuple(),
                (a_par, b_par) in Tensor::<CpuParBackend>::arbitrary_matmul_tuple(),
            ) {
                binary_grad_assert(a_seq, b_seq, |t1, t2| t1.mm(t2));
                binary_grad_assert(a_par, b_par, |t1, t2| t1.mm(t2));
            }

            #[test]
            fn permute_grad_tests(
                (t_seq, o_seq) in Tensor::<CpuSeqBackend>::arbitrary_with_order(),
                (t_par, o_par) in Tensor::<CpuParBackend>::arbitrary_with_order(),
            ) {
                permute_grad_test(t_seq, o_seq);
                permute_grad_test(t_par, o_par);
            }

            #[test]
            fn reduce_grad_tests(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                reduce_grad_test(t_seq);
                reduce_grad_test(t_par);
            }

            #[test]
            fn binary_grad_tests(
                (t1_seq, t2_seq) in Tensor::<CpuSeqBackend>::arbitrary_tuple(),
                (t1_par, t2_par) in Tensor::<CpuParBackend>::arbitrary_tuple(),
            ) {
                binary_grad_test(t1_seq, t2_seq);
                binary_grad_test(t1_par, t2_par);
            }

            // central diff doesn't work for lt and gt if x = y
            #[test]
            fn binary_grad_lt_gt_tests(
                (t1_seq, t2_seq) in Tensor::<CpuSeqBackend>::arbitrary_disjoint_tuple(),
                (t1_par, t2_par) in Tensor::<CpuParBackend>::arbitrary_disjoint_tuple(),
            ) {
                binary_grad_assert(t1_seq.clone(), t2_seq.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_seq.clone(), t2_seq.clone(), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1_par.clone(), t2_par.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_par.clone(), t2_par.clone(), |t1, t2| t1.lt(t2));
            }

            #[test]
            fn binary_grad_broadcast_tests(
                (t1_seq, t2_seq) in Tensor::<CpuSeqBackend>::arbitrary_tuple(),
                (t1_par, t2_par) in Tensor::<CpuParBackend>::arbitrary_tuple(),
            ) {
                binary_grad_broadcast_test(t1_seq, t2_seq);
                binary_grad_broadcast_test(t1_par, t2_par);
            }

            // central diff doesn't work for lt and gt if x = y
            #[test]
            fn binary_grad_broadcast_lt_gt_tests(
                (t1_seq, t2_seq) in Tensor::<CpuSeqBackend>::arbitrary_disjoint_tuple(),
                (t1_par, t2_par) in Tensor::<CpuParBackend>::arbitrary_disjoint_tuple(),
            ) {
                binary_grad_assert(t1_seq.clone().sum(Some(0)), t2_seq.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_seq.clone().sum(Some(0)), t2_seq.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_seq.clone(), t2_seq.clone().sum(Some(0)), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_seq.clone().sum(Some(0)), t2_seq.clone(), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1_seq.clone(), t2_seq.clone().sum(Some(0)), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1_par.clone().sum(Some(0)), t2_par.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_par.clone().sum(Some(0)), t2_par.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_par.clone(), t2_par.clone().sum(Some(0)), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1_par.clone().sum(Some(0)), t2_par.clone(), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1_par.clone(), t2_par.clone().sum(Some(0)), |t1, t2| t1.lt(t2));
            }

            #[test]
            fn unary_grad_complex_test1(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                unary_grad_complex_test1_(t_seq);
                unary_grad_complex_test1_(t_par);
            }

            // no zero since relu
            #[test]
            fn unary_grad_complex_test2(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary_no_zero(),
                t_par in Tensor::<CpuParBackend>::arbitrary_no_zero(),
            ) {
                unary_grad_complex_test2_(t_seq);
                unary_grad_complex_test2_(t_par);
            }

            #[test]
            fn unary_grad_relu_test(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary_no_zero(),
                t_par in Tensor::<CpuParBackend>::arbitrary_no_zero(),
            ) {
                unary_grad_assert(t_seq, |t| t.relu());
                unary_grad_assert(t_par, |t| t.relu());
            }

            #[test]
            fn unary_grad_tests(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                unary_grad_test(t_seq);
                unary_grad_test(t_par);
            }

            #[test]
            fn binary_tests(
                (t1_seq, t2_seq) in Tensor::<CpuSeqBackend>::arbitrary_tuple(),
                (t1_par, t2_par) in Tensor::<CpuParBackend>::arbitrary_tuple(),
            ) {
                binary_test(t1_seq, t2_seq);
                binary_test(t1_par, t2_par);
            }

            #[test]
            fn unary_tests(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                unary_test(t_seq);
                unary_test(t_par);
            }

            #[test]
            fn unary_complex_test1(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                unary_complex_test1_(t_seq);
                unary_complex_test1_(t_par);
            }

            #[test]
            fn unary_complex_test2(
                t_seq in Tensor::<CpuSeqBackend>::arbitrary(),
                t_par in Tensor::<CpuParBackend>::arbitrary(),
            ) {
                unary_complex_test2_(t_seq);
                unary_complex_test2_(t_par);
            }
        }

        fn reduce_forward_one_dim_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
        {
            let summed = t.sum(Some(0));
            let exp = Tensor::from_1d(&[11., 16.].map(B::Element::fromf));
            let is_close = summed.is_close(exp);
            let shape = Shape::scalar(is_close.size());
            assert_eq!(
                Some(B::Element::one()),
                is_close.view(&shape).all(Some(0)).item()
            );
        }

        #[test]
        fn test_reduce_forward_one_dim() {
            let shape = Shape::new(vec![3, 2]);
            let strides = (&shape).into();
            let td = CpuData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);

            let t_seq: Tensor<CpuSeqBackend> = Tensor::from_data(td.clone());
            reduce_forward_one_dim_test(t_seq);
            let t_par: Tensor<CpuParBackend> = Tensor::from_data(td.clone());
            reduce_forward_one_dim_test(t_par);
        }

        fn reduce_forward_one_dim_2_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
        {
            let summed = t.sum(Some(1));
            let exp = Tensor::from_2d(&[
                &[B::Element::fromf(5.)],
                &[B::Element::fromf(10.)],
                &[B::Element::fromf(12.)],
            ])
            .unwrap();
            let is_close = summed.is_close(exp);
            let shape = Shape::scalar(is_close.size());
            assert_eq!(
                Some(B::Element::one()),
                is_close.view(&shape).all(Some(0)).item()
            );
        }

        #[test]
        fn test_reduce_forward_one_dim_2() {
            let shape = Shape::new(vec![3, 2]);
            let strides = (&shape).into();
            let td = CpuData::new(vec![2., 3., 4., 6., 5., 7.], shape, strides);

            let t_seq: Tensor<CpuSeqBackend> = Tensor::from_data(td.clone());
            reduce_forward_one_dim_2_test(t_seq);
            let t_par: Tensor<CpuParBackend> = Tensor::from_data(td);
            reduce_forward_one_dim_2_test(t_par);
        }

        fn reduce_forward_all_dim_test<'a, B>(t: Tensor<'a, B>)
        where
            B: Backend,
        {
            let summed = t.sum(None);
            assert_eq!(Some(B::Element::fromf(27.)), summed.item());
        }

        #[test]
        fn test_reduce_forward_all_dim() {
            let shape = Shape::new(vec![3, 2]);
            let t_seq =
                Tensor::<CpuSeqBackend>::from_1d(&[2., 3., 4., 6., 5., 7.]).reshape(shape.clone());
            reduce_forward_all_dim_test(t_seq);
            let t_par = Tensor::<CpuParBackend>::from_1d(&[2., 3., 4., 6., 5., 7.]).reshape(shape);
            reduce_forward_all_dim_test(t_par);
        }
    }

    mod gpu {
        use crate::backend::backend::CpuSeqBackend;

        use super::*;

        fn unary_grad_assert<'a, F>(tensor: Tensor<'a, GpuBackend>, f: F)
        where
            F: Fn(Tensor<'a, GpuBackend>) -> Tensor<'a, GpuBackend>,
        {
            let reset = tensor.grad(None).trace(Trace::default());
            let idx = reset.data.shape().sample_idx();
            let out = f(reset.clone());
            let grads = out.sum(None).backward();
            let grad = grads.wrt(&reset);
            let grad_cpu = grad.unwrap().data.to_cpu().unwrap();
            let grad_strides = grad.unwrap().data.strides.clone();
            let grad_data = grad_cpu[grad_strides.position(&idx)];
            let check = unary_grad_central_diff(reset.clone(), f, &idx, 1e-3);
            assert!(
                grad_data.is_close(check),
                "tensor grad ({grad:?}) should be close to central diff ({check:?})",
            );
        }

        pub fn binary_grad_assert<'a, F>(
            tensor1: Tensor<'a, GpuBackend>,
            tensor2: Tensor<'a, GpuBackend>,
            f: F,
        ) where
            F: Fn(Tensor<'a, GpuBackend>, Tensor<'a, GpuBackend>) -> Tensor<'a, GpuBackend>,
        {
            let (reset1, reset2) = (
                tensor1.grad(None).trace(Trace::default()),
                tensor2.grad(None).trace(Trace::default()),
            );
            let (idx1, idx2) = (
                reset1.data.shape().sample_idx(),
                reset2.data.shape().sample_idx(),
            );
            let out = f(reset1.clone(), reset2.clone());
            let grads = out.sum(None).backward();
            let (grad1, grad2) = (grads.wrt(&reset1).unwrap(), grads.wrt(&reset2).unwrap());

            let (check1, check2) = (
                binary_grad_central_diff(reset1.clone(), reset2.clone(), &f, &idx1, true, 1e-3),
                binary_grad_central_diff(reset1.clone(), reset2.clone(), f, &idx2, false, 1e-3),
            );
            let (grad1_cpu, grad2_cpu) = (grad1.data.to_cpu().unwrap(), grad2.data.to_cpu().unwrap());
            let (grad1_strides, grad2_strides) =
                (grad1.data.strides.clone(), grad2.data.strides.clone());
            let (grad_data1, grad_data2) = (
                grad1_cpu[grad1_strides.position(&idx1)],
                grad2_cpu[grad2_strides.position(&idx2)],
            );
            assert!(
                grad_data1.is_close(check1),
                "tensor 1 grad ({grad1:?}) should be close to central diff ({check1:?})",
            );
            assert!(
                grad_data2.is_close(check2),
                "tensor 2 grad ({grad2:?}) should be close to central diff ({check2:?})",
            );
        }

        fn unary_assert<'a, FT, FF>(t: Tensor<'a, GpuBackend>, ft: FT, ff: FF)
        where
            FT: Fn(Tensor<'a, GpuBackend>) -> Tensor<'a, GpuBackend>,
            FF: Fn(f32) -> f32,
        {
            let data = t.data.to_cpu().unwrap();
            let strides = t.data.strides.clone();
            let res = ft(t);
            let res_data = res.data.to_cpu().unwrap();
            let res_strides = res.data.strides.clone();
            for idx in res.data.indices() {
                assert!(
                    res_data[res_strides.position(&idx)].is_close(ff(data[strides.position(&idx)]))
                );
            }
        }

        fn binary_assert<'a, FT, FF>(
            t1: Tensor<'a, GpuBackend>,
            t2: Tensor<'a, GpuBackend>,
            ft: FT,
            ff: FF,
        ) where
            FT: Fn(Tensor<'a, GpuBackend>, Tensor<'a, GpuBackend>) -> Tensor<'a, GpuBackend>,
            FF: Fn(f32, f32) -> f32,
        {
            let data1 = t1.data.to_cpu().unwrap();
            let strides1 = t1.data.strides.clone();

            let data2 = t2.data.to_cpu().unwrap();
            let strides2 = t2.data.strides.clone();

            let res = ft(t1, t2);
            let res_data = res.data.to_cpu().unwrap();
            let res_strides = res.data.strides.clone();

            for idx in res.data.indices() {
                assert!(res_data[res_strides.position(&idx)].is_close(ff(
                    data1[strides1.position(&idx)],
                    data2[strides2.position(&idx)]
                )));
            }
        }

        proptest! {
            #[test]
            #[serial(gpu)]
            fn matmul_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_matmul_tuple(),
            ) {
                if let [d, a, b] = t1.data.shape().data() && let [_, _, c] = t2.data.shape().data() {
                    let c_mm = t1.clone().mm(t2.clone());
                    let c_prime = (
                        t1.clone().contiguous().view(&Shape::new(vec![*d, *a, *b, 1])) *
                        t2.clone().contiguous().view(&Shape::new(vec![1, 1, *b, *c]))
                    ).sum(Some(2)).view(&Shape::new(vec![*d, *a, *c]));
                    assert_eq!(Some(1.), c_mm.is_close(c_prime).all(None).item());
                } else {
                    panic!("both tensors should be 3d");
                }
            }

            #[test]
            #[serial(gpu)]
            fn matmul_grad_tests(
                (a, b) in Tensor::<GpuBackend>::arbitrary_matmul_tuple(),
            ) {
                binary_grad_assert(a.clone(), b.clone(), |t1, t2| t1.mm(t2));
            }

            #[test]
            #[serial(gpu)]
            fn permute_grad_tests(
                (t, o) in Tensor::<GpuBackend>::arbitrary_with_order(),
            ) {
                unary_grad_assert(t, move |t| t.permute(o.clone()));
            }

            #[test]
            #[serial(gpu)]
            fn reduce_grad_tests(
                t in Tensor::<GpuBackend>::arbitrary(),
            ) {
                unary_grad_assert(t.clone(), |t| t.sum(Some(0)));
                unary_grad_assert(t.clone(), |t| t.mean(Some(0)));
                unary_grad_assert(t.clone(), |t| t.mean(None));
            }

            #[test]
            #[serial(gpu)]
            fn binary_grad_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_tuple(),
            ) {
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 + t2);
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 - t2);
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1 * t2);
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| {
                    t1 / (t2 + Tensor::from_scalar(5.5))
                });
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.eq(t2));
            }

            // central diff doesn't work for lt gt if x == y
            #[test]
            #[serial(gpu)]
            fn binary_grad_lt_gt_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_disjoint_tuple(),
            ) {
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1.clone(), t2.clone(), |t1, t2| t1.gt(t2));
            }

            #[test]
            #[serial(gpu)]
            fn binary_grad_broadcast_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_tuple(),
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
                binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.eq(t2));
                binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.eq(t2));
            }

            // central diff doesn't work for lt gt if x == y
            #[test]
            #[serial(gpu)]
            fn binary_grad_broadcast_lt_gt_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_disjoint_tuple(),
            ) {
                binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.lt(t2));
                binary_grad_assert(t1.clone().sum(Some(0)), t2.clone(), |t1, t2| t1.gt(t2));
                binary_grad_assert(t1.clone(), t2.clone().sum(Some(0)), |t1, t2| t1.gt(t2));
            }

            #[test]
            #[serial(gpu)]
            fn unary_grad_complex_test1(
                t in Tensor::<GpuBackend>::arbitrary(),
            ) {
                unary_grad_assert(t, |t|
                    (t.clone() + Tensor::from_scalar(100000.)).ln() +
                        (t - Tensor::from_scalar(200.)).exp()
                );
            }

            #[test]
            #[serial(gpu)]
            fn unary_grad_complex_test2(
                t in Tensor::<GpuBackend>::arbitrary_no_zero(),
            ) {
                unary_grad_assert(t, |t| {
                    ((((t * Tensor::from_scalar(10.) + Tensor::from_scalar(7.)).relu()
                        * Tensor::from_scalar(6.)
                        + Tensor::from_scalar(5.))
                    .relu()
                        * Tensor::from_scalar(10.))
                    .sig())
                    .ln()
                        / Tensor::from_scalar(50.)
                });
            }

            #[test]
            #[serial(gpu)]
            fn unary_grad_relu_test(
                t in Tensor::<GpuBackend>::arbitrary_no_zero(),
            ) {
                unary_grad_assert(t, |t| t.relu());
            }

            #[test]
            #[serial(gpu)]
            fn unary_grad_tests(t in Tensor::<GpuBackend>::arbitrary()) {
                unary_grad_assert(t.clone(), |t| -t);
                unary_grad_assert(t.clone(), |t| t.clone() * t);
                unary_grad_assert(t.clone(), |t| t.clone() * t.clone() * t);
                unary_grad_assert(t.clone(), |t| (t + Tensor::from_scalar(3.5)).inv());
                unary_grad_assert(t.clone(), |t| t.sig());
                unary_grad_assert(t.clone(), |t| t.exp());
                unary_grad_assert(t.clone(), |t| (t + Tensor::from_scalar(100000.)).ln());
            }

            #[test]
            #[serial(gpu)]
            fn binary_tests(
                (t1, t2) in Tensor::<GpuBackend>::arbitrary_tuple(),
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

            #[test]
            #[serial(gpu)]
            fn unary_tests(
                t in Tensor::<GpuBackend>::arbitrary(),
            ) {
                unary_assert(t.clone(), |t| -t, |f| -f);
                unary_assert(
                    t.clone(),
                    |t| t.inv(),
                    |f| {
                        if f != 0. {
                            1. / f
                        } else {
                            0.
                        }
                    },
                );
                unary_assert(t.clone(), |t| t.sig(), |f| f.sig());
                unary_assert(
                    t.clone(),
                    |t| t.ln(),
                    |f| if f > 0. { f.ln() } else { 0. },
                );
                unary_assert(t.clone(), |t| t.exp(), |f| f.exp());
                unary_assert(t.clone(), |t| t.relu(), |f| f.relu());
                unary_assert(t.clone(), |t| t.contiguous(), |f| f);
                unary_assert(t.clone(), |t| t.clone() * t, |f| f * f);
                unary_assert(t.clone(), |t| t.clone() * t.clone() * t, |f| f * f * f);
            }

            #[test]
            #[serial(gpu)]
            fn unary_complex_test1(
                t in Tensor::<GpuBackend>::arbitrary(),
            ) {
                let ff = |f: f32| (f + 100000.).ln() + (f - 200.).exp();
                unary_assert(t.clone(), |t| {
                    (t.clone() + Tensor::from_scalar(100000.)).ln() +
                        (t - Tensor::from_scalar(200.)).exp()
                }, ff);
            }

            #[test]
            #[serial(gpu)]
            fn unary_complex_test2(
                t in Tensor::<GpuBackend>::arbitrary(),
            ) {
                let ff =
                    |f: f32| ((((f * 10. + 7.).relu() * 6. + 5.).relu() * 10.).sig()).ln() / 50.;
                unary_assert(t.clone(), |t| {
                    ((((t * Tensor::from_scalar(10.) + Tensor::from_scalar(7.)).relu()
                        * Tensor::from_scalar(6.)
                        + Tensor::from_scalar(5.))
                    .relu()
                        * Tensor::from_scalar(10.))
                    .sig())
                    .ln()
                        / Tensor::from_scalar(50.)
                }, ff);
            }
        }

        #[test]
        #[serial(gpu)]
        fn test_reduce_forward_one_dim() {
            let shape = Shape::new(vec![3, 2]);
            let strides = (&shape).into();
            let td = GpuData::new(
                &[2., 3., 4., 6., 5., 7.],
                shape,
                strides,
                get_wgpu_context(),
            );

            let t: Tensor<GpuBackend> = Tensor::from_data(td);
            let summed = t.sum(Some(0));

            let exp = Tensor::from_1d(&[11., 16.]);
            let is_close = summed.is_close(exp);
            let shape = Shape::scalar(is_close.size());
            assert_eq!(Some(1.), is_close.view(&shape).all(Some(0)).item());
        }

        #[test]
        #[serial(gpu)]
        fn test_reduce_forward_one_dim_2() {
            let shape = Shape::new(vec![3, 2]);
            let strides = (&shape).into();
            let td = GpuData::new(
                &[2., 3., 4., 6., 5., 7.],
                shape,
                strides,
                get_wgpu_context(),
            );

            let t: Tensor<GpuBackend> = Tensor::from_data(td);
            let summed = t.sum(Some(1));

            let exp = Tensor::from_2d(&[&[5.], &[10.], &[12.]]).unwrap();
            let is_close = summed.is_close(exp);
            let shape = Shape::scalar(is_close.size());
            assert_eq!(Some(1.), is_close.view(&shape).all(Some(0)).item());
        }

        #[test]
        #[serial(gpu)]
        fn test_reduce_forward_all_dim() {
            let shape = Shape::new(vec![3, 2]);
            let strides = (&shape).into();
            let td = GpuData::new(
                &[2., 3., 4., 6., 5., 7.],
                shape,
                strides,
                get_wgpu_context(),
            );

            let t: Tensor<GpuBackend> = Tensor::from_data(td);
            let summed = t.sum(None);
            assert_eq!(Some(27.), summed.item());
        }

        #[test]
        #[serial(gpu)]
        fn test_backward_gpu() {
            let shape = Shape::new(vec![3, 1]);
            let strides: Strides = (&shape).into();

            let tdg = GpuData::new(
                &[1., 2., 3.],
                shape.clone(),
                strides.clone(),
                get_wgpu_context(),
            );
            let g: Tensor<GpuBackend> = Tensor::from_data(tdg);
            let gs = g.clone().view(&Shape::new(vec![3])).sum(None);
            let gres = gs.backward();
            assert_eq!(vec![1., 1., 1.], gres.wrt(&g).unwrap().data.collect());

            let tdc = CpuData::new(vec![1., 2., 3.], shape.clone(), strides.clone());
            let c: Tensor<CpuSeqBackend> = Tensor::from_data(tdc);
            let cs = c.clone().view(&Shape::new(vec![3])).sum(None);
            let cres = cs.backward();
            assert_eq!(vec![1., 1., 1.], cres.wrt(&c).unwrap().data.collect());
        }

        #[test]
        #[serial(gpu)]
        fn test_view_backward() {
            let xc: Tensor<CpuSeqBackend> =
                Tensor::from_2d(&[&[1., 2., 3.], &[4., 5., 6.]]).unwrap();
            let xc_size = xc.size();
            let vc = xc.clone().view(&Shape::scalar(xc_size));
            let yc = vc.sum(None);
            let mc = yc.backward();
            let xcg = mc.wrt(&xc).unwrap().data.collect();
            assert_eq!(vec![1., 1., 1., 1., 1., 1.], xcg);

            let xg: Tensor<GpuBackend> = Tensor::from_2d(&[&[1., 2., 3.], &[4., 5., 6.]]).unwrap();
            let xg_size = xg.size();
            let vg = xg.clone().view(&Shape::scalar(xg_size));
            let yg = vg.sum(None);
            let mg = yg.backward();
            let xgg = mg.wrt(&xg).unwrap().data.collect();
            assert_eq!(vec![1., 1., 1., 1., 1., 1.], xgg);
        }

        #[test]
        #[serial(gpu)]
        fn test_broadcast_mul_backward() {
            let xc: Tensor<CpuSeqBackend> =
                Tensor::from_2d(&[&[1., 2., 3.], &[4., 5., 6.]]).unwrap();
            let oc = xc.clone() * Tensor::from_scalar(2.);
            let lc = oc.sum(None);
            let mc = lc.backward();
            let xcg = mc.wrt(&xc).unwrap().data.collect();
            assert_eq!(vec![2., 2., 2., 2., 2., 2.], xcg);

            let xg: Tensor<GpuBackend> = Tensor::from_2d(&[&[1., 2., 3.], &[4., 5., 6.]]).unwrap();
            let og = xg.clone() * Tensor::from_scalar(2.);
            let lg = og.sum(None);
            let mg = lg.backward();
            let xgg = mg.wrt(&xg).unwrap().data.collect();
            assert_eq!(vec![2., 2., 2., 2., 2., 2.], xgg);
        }

        #[test]
        #[serial(gpu)]
        fn test_broadcast_add_backward() {
            let xc: Tensor<CpuSeqBackend> = Tensor::from_1d(&[1., 2., 3.]);
            let oc = xc.clone() + Tensor::from_scalar(5.);
            let lc = oc.sum(None);
            let mc = lc.backward();
            let xcg = mc.wrt(&xc).unwrap().data.collect();
            assert_eq!(vec![1., 1., 1.], xcg);

            let xg: Tensor<GpuBackend> = Tensor::from_1d(&[1., 2., 3.]);
            let og = xg.clone() + Tensor::from_scalar(5.);
            let lg = og.sum(None);
            let mg = lg.backward();
            let xgg = mg.wrt(&xg).unwrap().data.collect();
            assert_eq!(vec![1., 1., 1.], xgg);
        }

        #[test]
        #[serial(gpu)]
        fn test_expand_backward() {
            let a_shape = Shape::new(vec![2, 1]);
            let b_shape = Shape::new(vec![2, 3]);

            let ac: Tensor<CpuSeqBackend> = Tensor::from_shape(&[1., 2.], a_shape.clone());
            let bc = Tensor::from_shape(&[10., 20., 30., 40., 50., 60.], b_shape.clone());
            let oc = ac.clone() + bc;
            let lc = oc.sum(None);
            let mc = lc.backward();
            let acg = mc.wrt(&ac).unwrap().data.collect();
            assert_eq!(vec![3., 3.], acg);

            let ag: Tensor<GpuBackend> = Tensor::from_shape(&[1., 2.], a_shape.clone());
            let bg = Tensor::from_shape(&[10., 20., 30., 40., 50., 60.], b_shape.clone());
            let og = ag.clone() + bg;
            let lg = og.sum(None);
            let mg = lg.backward();
            let agg = mg.wrt(&ag).unwrap().data.collect();
            assert_eq!(vec![3., 3.], agg);
        }

        #[test]
        #[serial(gpu)]
        fn test_matmul_backward() {
            let a_shape = Shape::new(vec![1, 5, 3]);
            let b_shape = Shape::new(vec![1, 3, 1]);
            let a_grad = vec![
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ];
            let b_grad = vec![30., 35., 40.];

            let ac: Tensor<CpuSeqBackend> = Tensor::from_shape(
                &(0..15).map(|i| i as f64).collect::<Vec<_>>(),
                a_shape.clone(),
            );
            let bc = Tensor::from_shape(&[1., 2., 3.], b_shape.clone());
            let oc = ac.clone().mm(bc.clone());
            let lc = oc.sum(None);
            let mc = lc.backward();
            let acg = mc.wrt(&ac).unwrap().data.collect();
            assert_eq!(a_grad.clone(), acg);
            let bcg = mc.wrt(&bc).unwrap().data.collect();
            assert_eq!(b_grad.clone(), bcg);

            let ag: Tensor<GpuBackend> = Tensor::from_shape(
                &(0..15).map(|i| i as f32).collect::<Vec<_>>(),
                a_shape.clone(),
            );
            let bg = Tensor::from_shape(&[1., 2., 3., 4., 5., 6.], b_shape.clone());
            let og = ag.clone().mm(bg.clone());
            let lg = og.sum(None);
            let mg = lg.backward();
            let agg = mg.wrt(&ag).unwrap().data.collect();
            assert_eq!(a_grad.iter().map(|&f| f as f32).collect::<Vec<_>>(), agg);
            let bgg = mg.wrt(&bg).unwrap().data.collect();
            assert_eq!(b_grad.iter().map(|&f| f as f32).collect::<Vec<_>>(), bgg);
        }

        #[test]
        #[serial(gpu)]
        fn test_relu_backward() {
            let xc: Tensor<CpuSeqBackend> = Tensor::from_1d(&[-1., 0., 1.]);
            let oc = xc.clone().relu();
            let lc = oc.sum(None);
            let mc = lc.backward();
            let xcg = mc.wrt(&xc).unwrap().data.collect();
            assert_eq!(vec![0., 0., 1.], xcg);

            let xg: Tensor<GpuBackend> = Tensor::from_1d(&[-1., 0., 1.]);
            let og = xg.clone().relu();
            let lg = og.sum(None);
            let mg = lg.backward();
            let xgg = mg.wrt(&xg).unwrap().data.collect();
            assert_eq!(vec![0., 0., 1.], xgg);
        }

        #[test]
        #[serial(gpu)]
        fn test_sig_backward() {
            let xc: Tensor<CpuSeqBackend> = Tensor::from_1d(&[-1., 0., 1.]);
            let oc = xc.clone().sig();
            let lc = oc.sum(None);
            let mc = lc.backward();
            let xcg = mc.wrt(&xc).unwrap().data.collect();
            assert_eq!(vec![0.19661193324148185, 0.25, 0.19661193324148185], xcg);

            let xg: Tensor<GpuBackend> = Tensor::from_1d(&[-1., 0., 1.]);
            let og = xg.clone().sig();
            let lg = og.sum(None);
            let mg = lg.backward();
            let xgg = mg.wrt(&xg).unwrap().data.collect();
            assert_eq!(vec![0.19661194, 0.25, 0.19661193], xgg);
        }
    }

    fn view_test<'a, B>(t: Tensor<'a, B>)
    where
        B: Backend,
    {
        assert_eq!(&Shape::new(vec![2, 3]), t.data.shape());
        let t2 = t.clone().view(&Shape::new(vec![6]));
        assert_eq!(&Shape::new(vec![6]), t2.data.shape());
        let t3 = t2.view(&Shape::new(vec![1, 6]));
        assert_eq!(&Shape::new(vec![1, 6]), t3.data.shape());
        let t4 = t3.view(&Shape::new(vec![6, 1]));
        assert_eq!(&Shape::new(vec![6, 1]), t4.data.shape());
        let t5 = t4.view(&Shape::new(vec![2, 3]));
        assert_eq!(&Shape::new(vec![2, 3]), t5.data.shape());
        assert_eq!(Some(B::Element::one()), t.is_close(t5).all(None).item());
    }

    #[test]
    #[serial(gpu)]
    fn test_view() {
        let t_seq = Tensor::<CpuSeqBackend>::from_2d(&[&[2., 3., 4.], &[4., 5., 7.]]).unwrap();
        view_test(t_seq);
        let t_par = Tensor::<CpuParBackend>::from_2d(&[&[2., 3., 4.], &[4., 5., 7.]]).unwrap();
        view_test(t_par);
        let t_gpu = Tensor::<GpuBackend>::from_2d(&[&[2., 3., 4.], &[4., 5., 7.]]).unwrap();
        view_test(t_gpu);
    }
}
