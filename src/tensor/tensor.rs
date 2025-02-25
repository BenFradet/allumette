use crate::autodiff::history::History;
use std::ops;

use super::{
    forward::Forward,
    ops::{
        binary_ops::{Add, All, Eq, IsClose, Lt, Mul, Permute, Sum, View},
        unary_ops::{Exp, Inv, Ln, Neg, Relu, Sig},
    },
    shaping::{order::Order, shape::Shape},
    tensor_data::TensorData,
};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: TensorData,
    pub history: History<TensorData>,
}

impl Tensor {
    pub fn new(data: TensorData, history: History<TensorData>) -> Self {
        Self { data, history }
    }

    pub fn from_tensor_data(data: TensorData) -> Self {
        Self {
            data,
            history: History::default(),
        }
    }

    pub fn lt(self, rhs: Tensor) -> Self {
        Forward::binary(Lt {}, self.data, rhs.data)
    }

    pub fn gt(self, rhs: Tensor) -> Self {
        Forward::binary(Lt {}, rhs.data, self.data)
    }

    pub fn eq(self, rhs: Tensor) -> Self {
        Forward::binary(Eq {}, self.data, rhs.data)
    }

    pub fn all(self, dim: usize) -> Self {
        Forward::binary(All {}, self.data, TensorData::scalar(dim as f64))
    }

    pub fn sum(self, dim: usize) -> Self {
        Forward::binary(Sum {}, self.data, TensorData::scalar(dim as f64))
    }

    pub fn mean(self, dim: usize) -> Self {
        self.sum(dim) / Self::from_tensor_data(TensorData::scalar(dim as f64))
    }

    pub fn permute(self, order: Order) -> Self {
        let fs = order.data.iter().map(|u| *u as f64).collect();
        let td = TensorData::vec(fs);
        Forward::binary(Permute {}, self.data, td)
    }

    pub fn view(self, shape: Shape) -> Self {
        let fs = shape.data().iter().map(|u| *u as f64).collect();
        let td = TensorData::vec(fs);
        Forward::binary(View {}, self.data, td)
    }

    pub fn is_close(self, rhs: Tensor) -> Self {
        Forward::binary(IsClose {}, self.data, rhs.data)
    }

    pub fn sigmoid(self) -> Self {
        Forward::unary(Sig {}, self.data)
    }

    pub fn relu(self) -> Self {
        Forward::unary(Relu {}, self.data)
    }

    pub fn ln(self) -> Self {
        Forward::unary(Ln {}, self.data)
    }

    pub fn exp(self) -> Self {
        Forward::unary(Exp {}, self.data)
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        Forward::binary(Add {}, self.data, rhs.data)
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs.data);
        Forward::binary(Add {}, self.data, new_rhs.data)
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Forward::binary(Mul {}, self.data, rhs.data)
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let new_rhs = Forward::unary(Inv {}, rhs.data);
        Forward::binary(Mul {}, self.data, new_rhs.data)
    }
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self.data)
    }
}
