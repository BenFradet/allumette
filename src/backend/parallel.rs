use std::sync::Arc;

use rayon::prelude::*;

use crate::{
    data::cpu_tensor_data::CpuTensorData,
    shaping::{shape::Shape, strides::Strides},
};

use super::{backend::Backend, backend_type::Par};

impl Backend<Par> for CpuTensorData {
    fn map<F: Fn(f64) -> f64 + Sync>(&self, f: F) -> Self {
        let out: Vec<_> = self.data.par_iter().map(|d| f(*d)).collect();
        Self {
            data: Arc::new(out),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    // TODO: remove unwrap
    fn map_broadcast<F: Fn(f64) -> f64 + Sync>(&self, out: &Self, f: F) -> Option<Self> {
        let len = out.shape.size;
        let strides: Strides = (&out.shape).into();
        let out_vec: Vec<_> = if self.shape == out.shape {
            (0..len).into_par_iter().map(|i| f(self.data[i])).collect()
        } else {
            (0..len)
                .into_par_iter()
                .map(|i| {
                    let out_idx = strides.idx(i);
                    let idx_bc = out_idx.broadcast(&self.shape).unwrap();
                    let pos_in = self.strides.position(&idx_bc);
                    f(self.data[pos_in])
                })
                .collect()
        };
        Some(Self::new(out_vec, out.shape.clone(), strides))
    }

    // TODO: remove unwrap
    fn zip<F: Fn(f64, f64) -> f64 + Sync>(&self, other: &Self, f: F) -> Option<Self> {
        if self.shape == other.shape {
            let len = self.shape.size;
            let out = (0..len)
                .into_par_iter()
                .map(|i| f(self.data[i], other.data[i]))
                .collect();
            Some(Self::new(out, self.shape.clone(), self.strides.clone()))
        } else {
            let shape = self.shape.broadcast(&other.shape)?;
            let strides: Strides = (&shape).into();
            let len = shape.size;
            let out = (0..len)
                .into_par_iter()
                .map(|i| {
                    let idx = strides.idx(i);
                    let idxa = idx.broadcast(&self.shape).unwrap();
                    let idxb = idx.broadcast(&other.shape).unwrap();
                    let posa = self.strides.position(&idxa);
                    let posb = other.strides.position(&idxb);
                    f(self.data[posa], other.data[posb])
                })
                .collect();
            Some(Self::new(out, shape, strides))
        }
    }

    fn reduce<F: Fn(f64, f64) -> f64 + Sync>(&self, f: F, dim: usize, init: f64) -> Option<Self> {
        if dim < self.shape.data().len() {
            let mut shape_data = self.shape.data().to_vec();
            shape_data[dim] = 1;
            let shape = Shape::new(shape_data);
            let strides: Strides = (&shape).into();
            let len = shape.size;

            let out = (0..len)
                .into_par_iter()
                .map(|i| {
                    let out_idx = strides.idx(i);
                    let mut tmp = init;
                    for j in 0..self.shape[dim] {
                        let mut self_idx = out_idx.clone();
                        self_idx[dim] = j;
                        let self_pos = self.strides.position(&self_idx);
                        tmp = f(tmp, self.data[self_pos]);
                    }
                    tmp
                })
                .collect();
            Some(Self::new(out, shape, strides))
        } else {
            None
        }
    }
}
