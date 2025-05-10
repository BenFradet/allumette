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

    fn map_broadcast<F: Fn(f64) -> f64 + Sync>(&self, out: &Self, f: F) -> Option<Self> {
        let len = out.shape.size;
        let strides: Strides = (&out.shape).into();
        let out_vec: Vec<_> = if self.shape == out.shape {
            (0..len).into_par_iter().map(|i| f(self.data[i])).collect()
        } else {
            (0..len).into_par_iter().map(|i| {
                let out_idx = strides.idx(i);
                let idx_bc = out_idx.broadcast(&self.shape).unwrap();
                let pos_in = self.strides.position(&idx_bc);
                f(self.data[pos_in])
            }).collect()
        };
        Some(Self::new(out_vec, out.shape.clone(), strides))
    }

    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Option<Self> {
        let shape = if self.shape == other.shape {
            self.shape.clone()
        } else {
            self.shape.broadcast(&other.shape)?
        };
        let strides: Strides = (&shape).into();
        let len = shape.size;
        let mut out = vec![0.; len];
        for i in 0..len {
            let idx = strides.idx(i);
            let idxa = idx.broadcast(&self.shape)?;
            let idxb = idx.broadcast(&other.shape)?;
            let posa = self.strides.position(&idxa);
            let posb = other.strides.position(&idxb);
            let va = self.data[posa];
            let vb = other.data[posb];
            let pos = strides.position(&idx);
            out[pos] = f(va, vb);
        }
        Some(Self::new(out, shape, strides))
    }

    fn reduce(&self, f: impl Fn(f64, f64) -> f64, dim: usize, init: f64) -> Option<Self> {
        if dim < self.shape.data().len() {
            let mut shape_data = self.shape.data().to_vec();
            shape_data[dim] = 1;
            let shape = Shape::new(shape_data);
            let strides: Strides = (&shape).into();
            let len = shape.size;
            let mut out = vec![init; len];
            for i in 0..len {
                let out_idx = strides.idx(i);
                let out_pos = strides.position(&out_idx);
                for j in 0..self.shape[dim] {
                    let mut self_idx = out_idx.clone();
                    self_idx[dim] = j;
                    let self_pos = self.strides.position(&self_idx);
                    let v_out = out[out_pos];
                    let v_self = self.data[self_pos];
                    out[out_pos] = f(v_out, v_self);
                }
            }
            Some(Self::new(out, shape, strides))
        } else {
            None
        }
    }
}
