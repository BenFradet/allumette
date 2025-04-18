use std::{ops::Index, sync::Arc};

use proptest::{collection, prelude::*};

use super::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

#[derive(Clone, Debug)]
pub struct TensorData {
    pub data: Arc<Vec<f64>>,
    pub shape: Shape,
    pub strides: Strides,
}

impl TensorData {
    // TODO: accept any collection
    pub fn new(data: Vec<f64>, shape: Shape, strides: Strides) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn scalar(data: f64) -> Self {
        let shape = Shape::new(vec![1]);
        let strides = (&shape).into();
        Self {
            data: Arc::new(vec![data]),
            shape,
            strides,
        }
    }

    // TODO: accept any collection
    pub fn vec(data: Vec<f64>) -> Option<Self> {
        if data.is_empty() {
            None
        } else {
            let shape = Shape::new(vec![data.len()]);
            let strides = (&shape).into();
            Some(Self {
                data: Arc::new(data),
                shape,
                strides,
            })
        }
    }

    // TODO: accept any collection
    pub fn matrix(data: Vec<Vec<f64>>) -> Option<Self> {
        if data.is_empty() {
            None
        } else {
            let rows = data[0].len();
            if !data.iter().all(|v| v.len() == rows) {
                None
            } else {
                let cols = data.len();
                let shape = Shape::new(vec![cols, rows]);
                let strides = (&shape).into();
                Some(Self {
                    data: Arc::new(data.concat()),
                    shape,
                    strides,
                })
            }
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn epsilon(shape: Shape, idx: &Idx, eps: f64) -> Self {
        let strides: Strides = (&shape).into();
        let mut data = vec![0.; shape.size];
        data[strides.position(idx)] = eps;
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn rand(shape: Shape) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..shape.size).map(|_| rng.gen()).collect();
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        let strides = (&shape).into();
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
        }
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
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

    pub fn map_broadcast(&self, out: &TensorData, f: impl Fn(f64) -> f64) -> Option<Self> {
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

    pub fn zip(&self, other: &TensorData, f: impl Fn(f64, f64) -> f64) -> Option<TensorData> {
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

    pub fn reduce(&self, f: impl Fn(f64, f64) -> f64, dim: usize, init: f64) -> Option<Self> {
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

    // output of backward not the same size as the input of forward
    pub fn expand(&self, other: Self) -> Option<Self> {
        if self.shape == other.shape {
            return Some(other);
        }

        let bc_shape = self.shape.broadcast(&other.shape)?;
        let buf = TensorData::zeros(bc_shape);
        let mut out = other.map_broadcast(&buf, |f| f)?;
        if self.shape == out.shape {
            return Some(out);
        }

        let orig_shape = Shape::new(
            [
                vec![1; out.shape.len() - self.shape.len()],
                self.shape.data().to_vec(),
            ]
            .concat(),
        );
        for (dim, shape) in out.shape.clone().data().iter().enumerate() {
            if orig_shape.data()[dim] == 1 && *shape != 1 {
                out = out.reduce(|a, b| a + b, dim, 0.)?;
            }
        }
        assert!(
            out.size() == self.size(),
            "out shape: {:?}, self shape: {:?}",
            out.shape,
            self.shape
        );
        Some(out)
    }

    pub fn size(&self) -> usize {
        self.shape.size
    }

    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    pub fn indices(&self) -> impl Iterator<Item = Idx> + use<'_> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    pub fn permute(&self, order: &Order) -> Option<Self> {
        let n = self.shape.data().len();
        if order.fits(n) {
            let mut new_shape = vec![0; n];
            let mut new_strides = vec![0; n];
            for (idx, value) in order.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                new_strides[idx] = self.strides[value];
            }
            Some(Self {
                data: Arc::clone(&self.data),
                shape: Shape::new(new_shape),
                strides: Strides::new(new_strides),
            })
        } else {
            None
        }
    }

    pub fn is_contiguous(&self) -> bool {
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

    pub fn arbitrary() -> impl Strategy<Value = TensorData> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f64..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: Strides = (&shape).into();
                TensorData::new(data, shape, strides)
            })
    }
}

impl Index<Idx> for TensorData {
    type Output = f64;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.strides.position(&index)]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn repro_expand_test() {
        let input = TensorData::scalar(0.);
        let deriv = TensorData::vec(vec![1., 1.]).unwrap();
        let res = input.expand(deriv).map(|d| d.data).unwrap();
        assert_eq!(vec![2.], *res);
    }

    fn assert_tensor_eq(t1: &TensorData, t2: &TensorData) -> () {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.data, t2.data);
    }

    proptest! {
        #[test]
        fn reduce_test_sum(t1 in TensorData::arbitrary()) {
            let mut t1p = t1.clone();
            for i in 0..t1.shape.data().len() {
                t1p = t1p.reduce(|a, b| a + b, i, 0.).unwrap();
            }
            let res = t1.data.clone().iter().fold(0., |acc, a| acc + a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn reduce_test_mul(t1 in TensorData::arbitrary()) {
            let mut t1p = t1.clone();
            for i in 0..t1.shape.data().len() {
                t1p = t1p.reduce(|a, b| a * b, i, 1.).unwrap();
            }
            let res = t1.data.clone().iter().fold(1., |acc, a| acc * a);
            assert_eq!(1, t1p.data.len());
            assert!((res - t1p.data[0]).abs() < f64::EPSILON * 10_f64.powf(2.));
        }

        #[test]
        fn zip_commutative_test(t1 in TensorData::arbitrary(), t2 in TensorData::arbitrary()) {
            // this works if f is commutative
            let res1 = t1.zip(&t2, |a, b| a + b);
            let res2 = t2.zip(&t1, |a, b| a + b);
            match (res1, res2) {
                (Some(r1), Some(r2)) => assert_tensor_eq(&r1, &r2),
                (None, None) => (),
                (r1, r2) => panic!("{:?} not equal to {:?}", r1, r2),
            }
        }

        #[test]
        fn map_identity_test(t in TensorData::arbitrary()) {
            assert_tensor_eq(&t, &t.map(|f| f));
        }

        #[test]
        fn map_broadcast_identity_test(t in TensorData::arbitrary()) {
            let bc = t.map_broadcast(&t, |f| f);
            assert!(bc.is_some());
            assert_tensor_eq(&t, bc.as_ref().unwrap());
        }

        #[test]
        fn map_composition_test(t in TensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            assert_tensor_eq(&t.clone().map(f).map(g), &t.map(fg));
        }

        #[test]
        fn map_broadcast_composition_test(t in TensorData::arbitrary()) {
            let f = |a: f64| a * 2.;
            let g = |a: f64| a.powf(2.);
            let fg = |a: f64| g(f(a));
            let t1 = &t.clone().map_broadcast(&t, f).and_then(|t| t.map_broadcast(&t, g));
            let t2 = &t.map_broadcast(&t, fg);
            assert!(t1.is_some());
            assert!(t2.is_some());
            assert_tensor_eq(t1.as_ref().unwrap(), t2.as_ref().unwrap());
        }

        #[test]
        fn map_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let map = TensorData::zeros(shape.clone()).map(|z| z + f);
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }

        #[test]
        fn map_broadcast_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let t = TensorData::zeros(shape.clone());
            let res = t.map_broadcast(&t, |z| z + f);
            assert!(res.is_some());
            let map = res.unwrap();
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }

        #[test]
        fn zeros_test(shape in Shape::arbitrary()) {
            let zeros = TensorData::zeros(shape.clone());
            assert_eq!(shape.size, zeros.data.len());
            assert!(zeros.data.iter().all(|f| *f == 0.));
        }

        #[test]
        fn permute_test(tensor_data in TensorData::arbitrary(), idx in Idx::arbitrary()) {
            let reversed_index = idx.clone().reverse();
            let pos = tensor_data.strides.position(&idx);
            let order = Order::range(tensor_data.shape.data().len()).reverse();
            let perm_opt = tensor_data.permute(&order);
            assert!(perm_opt.is_some());
            let perm = perm_opt.unwrap();
            assert_eq!(pos, perm.strides.position(&reversed_index));
            let orig_opt = perm.permute(&order);
            assert!(orig_opt.is_some());
            let orig = orig_opt.unwrap();
            assert_eq!(pos, orig.strides.position(&idx));
        }

        #[test]
        fn enumeration_test(tensor_data in TensorData::arbitrary()) {
            let indices: Vec<_> = tensor_data.indices().collect();
            let count = indices.len();
            assert_eq!(tensor_data.size(), count);
            let set: HashSet<_> = indices.clone().into_iter().collect();
            assert_eq!(set.len(), count);
            for idx in indices {
                for (i, p) in idx.iter().enumerate() {
                    assert!(p < tensor_data.shape[i]);
                }
            }
        }
    }

    #[test]
    fn idx_in_set_test() -> () {
        let idx1 = Idx::new(vec![1, 2]);
        let idx2 = Idx::new(vec![1, 2]);
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![3, 5]);
        let strides = Strides::new(vec![5, 1]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new(vec![3, 5]), tensor.shape);
        assert_eq!(5, tensor.strides.position(&Idx::new(vec![1, 0])));
        assert_eq!(7, tensor.strides.position(&Idx::new(vec![1, 2])));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![5, 3]);
        let strides = Strides::new(vec![1, 5]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new(vec![5, 3]), tensor.shape);
    }

    #[test]
    fn rand_test() {
        let rand = TensorData::rand(Shape::new(vec![2]));
        println!("rand {:?}", rand);
        assert!(rand.data[0] != rand.data[1]);
    }
}
