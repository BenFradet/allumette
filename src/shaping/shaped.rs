use rand::Rng;

use crate::tensor::tensor_data::TensorData;
use std::{slice::Iter, sync::Arc};

use super::shape::Shape;

pub trait Shaped {
    fn shape(&self) -> &Shape;
    fn size(&self) -> usize;
    fn iter(&self) -> Iter<'_, f64>;
    fn first(&self) -> Option<f64>;
    fn is_contiguous(&self) -> bool;
    fn reshape(&self, shape: Shape) -> Self;

    fn ones(shape: Shape) -> Self;
    fn zeros(shape: Shape) -> Self;
    fn rand(shape: Shape) -> Self;

    fn scalar(s: f64) -> Self;
    fn vec(v: Vec<f64>) -> Self;
    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized;
}

impl Shaped for TensorData {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    fn iter(&self) -> Iter<'_, f64> {
        self.data.iter()
    }

    fn first(&self) -> Option<f64> {
        self.data.first().copied()
    }

    fn is_contiguous(&self) -> bool {
        if self.strides.is_empty() {
            false
        } else {
            let mut last = self.strides[0];
            for stride in self.strides.iter() {
                if stride > last {
                    return false;
                }
                last = stride;
            }
            true
        }
    }

    fn reshape(&self, shape: Shape) -> Self {
        let strides = (&shape).into();
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
        }
    }

    fn ones(shape: Shape) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn zeros(shape: Shape) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn rand(shape: Shape) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..shape.size).map(|_| rng.gen()).collect();
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn scalar(s: f64) -> Self {
        let shape = Shape::new(vec![1]);
        let strides = (&shape).into();
        Self {
            data: Arc::new(vec![s]),
            shape,
            strides,
        }
    }

    fn vec(v: Vec<f64>) -> Self {
        let shape = Shape::new(vec![v.len()]);
        let strides = (&shape).into();
        Self {
            data: Arc::new(v),
            shape,
            strides,
        }
    }

    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized,
    {
        if m.is_empty() {
            None
        } else {
            let rows = m[0].len();
            if !m.iter().all(|v| v.len() == rows) {
                None
            } else {
                let cols = m.len();
                let shape = Shape::new(vec![cols, rows]);
                let strides = (&shape).into();
                Some(Self {
                    data: Arc::new(m.concat()),
                    shape,
                    strides,
                })
            }
        }
    }
}
