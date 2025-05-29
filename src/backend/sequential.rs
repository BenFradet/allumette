use std::sync::Arc;

use crate::{
    data::{cpu_tensor_data::CpuTensorData, tensor_data::TensorData},
    shaping::{shape::Shape, strides::Strides},
};

use super::{backend::TensorBackend, backend_type::Seq};

impl TensorBackend<Seq> for CpuTensorData {
    fn map<F: Fn(f64) -> f64 + Sync>(&self, f: F) -> Self {
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

    fn map_broadcast<F: Fn(f64) -> f64 + Sync>(&self, out: &Self, f: F) -> Option<Self> {
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
        Some(Self::new(out_vec, out.shape.clone(), strides))
    }

    fn zip<F: Fn(f64, f64) -> f64 + Sync>(&self, other: &Self, f: F) -> Option<Self> {
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

    fn reduce<F: Fn(f64, f64) -> f64 + Sync>(&self, f: F, dim: usize, init: f64) -> Option<Self> {
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

#[cfg(test)]
mod tests {
    use proptest::proptest;

    use super::*;

    #[test]
    fn expand_test() {
        let input = CpuTensorData::scalar(0.);
        let deriv = CpuTensorData::vec(vec![1., 1.]);
        let res = TensorBackend::<Seq>::expand(&input, deriv)
            .map(|d| d.data)
            .unwrap();
        assert_eq!(vec![2.], *res);
    }

    fn assert_tensor_eq(t1: &CpuTensorData, t2: &CpuTensorData) {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.data, t2.data);
    }

    proptest! {
        #[test]
        fn reduce_test_sum(t1 in CpuTensorData::arbitrary()) {
            let mut t1p = t1.clone();
            for i in 0..t1.shape.data().len() {
                t1p = TensorBackend::<Seq>::reduce(&t1p, |a, b| a + b, i, 0.).unwrap();
            }
            let res = t1.data.clone().iter().fold(0., |acc, a| acc + a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn reduce_test_mul(t1 in CpuTensorData::arbitrary()) {
            let mut t1p = t1.clone();
            for i in 0..t1.shape.data().len() {
                t1p = TensorBackend::<Seq>::reduce(&t1p, |a, b| a * b, i, 1.).unwrap();
            }
            let res = t1.data.clone().iter().fold(1., |acc, a| acc * a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn zip_commutative_test(t1 in CpuTensorData::arbitrary(), t2 in CpuTensorData::arbitrary()) {
            // this works if f is commutative
            let res1 = TensorBackend::<Seq>::zip(&t1, &t2, |a, b| a + b);
            let res2 = TensorBackend::<Seq>::zip(&t2, &t1, |a, b| a + b);
            match (res1, res2) {
                (Some(r1), Some(r2)) => assert_tensor_eq(&r1, &r2),
                (None, None) => (),
                (r1, r2) => panic!("{r1:?} not equal to {r2:?}"),
            }
        }

        #[test]
        fn map_identity_test(t in CpuTensorData::arbitrary()) {
            assert_tensor_eq(&t, &TensorBackend::<Seq>::map(&t, |f| f));
        }

        #[test]
        fn map_broadcast_identity_test(t in CpuTensorData::arbitrary()) {
            let bc = TensorBackend::<Seq>::map_broadcast(&t, &t, |f| f);
            assert!(bc.is_some());
            assert_tensor_eq(&t, bc.as_ref().unwrap());
        }

        #[test]
        fn map_composition_test(t in CpuTensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            assert_tensor_eq(
                &TensorBackend::<Seq>::map(&TensorBackend::<Seq>::map(&t.clone(), f), g),
                &TensorBackend::<Seq>::map(&t, fg)
            );
        }

        #[test]
        fn map_broadcast_composition_test(t in CpuTensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            let t1 = TensorBackend::<Seq>::map_broadcast(&t.clone(), &t, f)
                .and_then(|t| TensorBackend::<Seq>::map_broadcast(&t, &t, g));
            let t2 = TensorBackend::<Seq>::map_broadcast(&t, &t, fg);
            assert!(t1.is_some());
            assert!(t2.is_some());
            assert_tensor_eq(t1.as_ref().unwrap(), t2.as_ref().unwrap());
        }

        #[test]
        fn map_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let map = TensorBackend::<Seq>::map(&CpuTensorData::zeros(shape.clone()), |z| z + f);
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }

        #[test]
        fn map_broadcast_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let t = CpuTensorData::zeros(shape.clone());
            let res = TensorBackend::<Seq>::map_broadcast(&t, &t, |z| z + f);
            assert!(res.is_some());
            let map = res.unwrap();
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }
    }
}
