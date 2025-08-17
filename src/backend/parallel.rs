use std::sync::Arc;

use rayon::prelude::*;

use crate::{
    data::cpu_tensor_data::CpuTensorData,
    shaping::{shape::Shape, strides::Strides},
};

use super::{backend::TensorBackend, backend_type::Par};

impl TensorBackend<f64, Par> for CpuTensorData {
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

    // TODO: look into https://github.com/rayon-rs/rayon/tree/main/rayon-demo/src/matmul
    fn matmul(&self, other: &Self) -> Self {
        let self_shape_len = self.shape.len();
        let other_shape_len = other.shape.len();
        assert!(self.shape[self_shape_len - 1] == other.shape[other_shape_len - 2]);

        let self_shape = self.shape.clone().drop_right(2);
        let other_shape = other.shape.clone().drop_right(2);

        let mut shape = self_shape.broadcast(&other_shape).unwrap();
        shape.push(self.shape[self_shape_len - 2]);
        shape.push(other.shape[other_shape_len - 1]);
        let len = shape.size;
        let strides: Strides = (&shape).into();

        let out: Vec<_> = (0..len)
            .into_par_iter()
            .map(|i| {
                let index = shape.idx(i);
                let mut self_idx = index.broadcast(&self.shape).unwrap();
                let self_idx_len = self_idx.len();
                let mut other_idx = index.broadcast(&other.shape).unwrap();
                let other_idx_len = other_idx.len();

                let mut tmp = 0.;
                for position in 0..self.shape[self_shape_len - 1] {
                    self_idx[self_idx_len - 1] = position;
                    other_idx[other_idx_len - 2] = position;
                    let self_pos = self.strides.position(&self_idx);
                    let other_pos = other.strides.position(&other_idx);
                    tmp += self.data[self_pos] * other.data[other_pos];
                }
                tmp
            })
            .collect();

        Self::new(out, shape, strides)
    }
}

#[cfg(test)]
mod tests {
    use proptest::proptest;

    use crate::backend::backend::Backend;

    use super::*;

    #[test]
    fn expand_test() {
        let input = CpuTensorData::from_scalar(0.);
        let deriv = CpuTensorData::from_1d(&[1., 1.]);
        let res = TensorBackend::<f64, Par>::expand(&input, deriv)
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
                t1p = TensorBackend::<f64, Par>::reduce(&t1p, |a, b| a + b, i, 0.).unwrap();
            }
            let res = t1.data.clone().iter().fold(0., |acc, a| acc + a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn reduce_test_mul(t1 in CpuTensorData::arbitrary()) {
            let mut t1p = t1.clone();
            for i in 0..t1.shape.data().len() {
                t1p = TensorBackend::<f64, Par>::reduce(&t1p, |a, b| a * b, i, 1.).unwrap();
            }
            let res = t1.data.clone().iter().fold(1., |acc, a| acc * a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn zip_commutative_test(t1 in CpuTensorData::arbitrary(), t2 in CpuTensorData::arbitrary()) {
            // this works if f is commutative
            let res1 = TensorBackend::<f64, Par>::zip(&t1, &t2, |a, b| a + b);
            let res2 = TensorBackend::<f64, Par>::zip(&t2, &t1, |a, b| a + b);
            match (res1, res2) {
                (Some(r1), Some(r2)) => assert_tensor_eq(&r1, &r2),
                (None, None) => (),
                (r1, r2) => panic!("{r1:?} not equal to {r2:?}"),
            }
        }

        #[test]
        fn map_identity_test(t in CpuTensorData::arbitrary()) {
            assert_tensor_eq(&t, &TensorBackend::<f64, Par>::map(&t, |f| f));
        }

        #[test]
        fn map_broadcast_identity_test(t in CpuTensorData::arbitrary()) {
            let bc = TensorBackend::<f64, Par>::map_broadcast(&t, &t, |f| f);
            assert!(bc.is_some());
            assert_tensor_eq(&t, bc.as_ref().unwrap());
        }

        #[test]
        fn map_composition_test(t in CpuTensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            assert_tensor_eq(
                &TensorBackend::<f64, Par>::map(&TensorBackend::<f64, Par>::map(&t.clone(), f), g),
                &TensorBackend::<f64, Par>::map(&t, fg)
            );
        }

        #[test]
        fn map_broadcast_composition_test(t in CpuTensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            let t1 = TensorBackend::<f64, Par>::map_broadcast(&t.clone(), &t, f)
                .and_then(|t| TensorBackend::<f64, Par>::map_broadcast(&t, &t, g));
            let t2 = TensorBackend::<f64, Par>::map_broadcast(&t, &t, fg);
            assert!(t1.is_some());
            assert!(t2.is_some());
            assert_tensor_eq(t1.as_ref().unwrap(), t2.as_ref().unwrap());
        }

        #[test]
        fn map_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let map = TensorBackend::<f64, Par>::map(&CpuTensorData::zeros(shape.clone()), |z| z + f);
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }

        #[test]
        fn map_broadcast_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let t = CpuTensorData::zeros(shape.clone());
            let res = TensorBackend::<f64, Par>::map_broadcast(&t, &t, |z| z + f);
            assert!(res.is_some());
            let map = res.unwrap();
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }
    }
}
