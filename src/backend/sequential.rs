use std::sync::Arc;

use crate::tensor::{
    shaping::{shape::Shape, strides::Strides},
    tensor_data::TensorData,
};

use super::{backend::Backend, backend_type::Sequential};

impl Backend<Sequential> for TensorData {
    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let len = self.size();
        let mut out = vec![0.; len];
        // TODO: add an iterator
        for (i, d) in self.data.iter().enumerate() {
            out[i] = f(*d);
        }
        Self {
            data: Arc::new(out),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    fn map_broadcast(&self, out: &Self, f: impl Fn(f64) -> f64) -> Option<Self> {
        let strides: Strides = (&out.shape).into();
        let len = out.shape.size;
        let mut out_vec = vec![0.; len];
        for i in 0..len {
            let out_idx = strides.idx(i);
            let idx_bc = out_idx.broadcast(&self.shape)?;
            let pos_in = self.strides.position(&idx_bc);
            let v = self.data[pos_in];
            let pos_out = out.strides.position(&out_idx);
            out_vec[pos_out] = f(v);
        }
        Some(TensorData::new(out_vec, out.shape.clone(), strides))
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
        Some(TensorData::new(out, shape, strides))
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
            Some(TensorData::new(out, shape, strides))
        } else {
            None
        }
    }
}
