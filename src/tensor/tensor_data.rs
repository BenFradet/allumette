use std::sync::Arc;

use proptest::{collection, prelude::*};

use super::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

#[derive(Clone, Debug)]
pub struct TensorData {
    pub data: Arc<Vec<f64>>,
    pub shape: Shape,
    pub strides: Strides,
}

impl TensorData {
    fn new(data: Vec<f64>, shape: Shape, strides: Strides) -> Self {
        TensorData {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        TensorData {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        TensorData {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn map(mut self, f: impl Fn(f64) -> f64) -> Self {
        let len = self.size();
        let mut out = vec![0.; len];
        // TODO: add an iterator
        for (i, d) in self.data.iter().enumerate() {
            out[i] = f(*d);
        }
        self.data = Arc::new(out);
        self
    }

    // feature(generic_const_exprs)
    pub fn zip(&self, other: &TensorData, f: impl Fn(f64, f64) -> f64) -> Option<TensorData> {
        let shape = self.shape.broadcast(&other.shape)?;
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

    pub fn reduce(&self, f: impl Fn(f64, f64) -> f64, dim: usize) -> TensorData {
        let mut shape_data = self.shape.data().to_vec();
        shape_data[dim] = 1;
        let shape = Shape::new(shape_data);
        let strides: Strides = (&shape).into();
        let len = shape.size;
        let mut out = vec![0.; len];
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
        TensorData::new(out, shape, strides)
    }

    pub fn size(&self) -> usize {
        self.shape.size
    }

    pub fn indices(&self) -> impl Iterator<Item = Idx> + use<'_> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    fn permute(mut self, order: &Order) -> Option<Self> {
        let n = self.shape.data().len();
        if order.fits(n) {
            let mut new_shape = vec![0; n];
            let mut new_strides = vec![0; n];
            for (idx, value) in order.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                new_strides[idx] = self.strides[value];
            }
            self.shape = Shape::new(new_shape);
            self.strides = Strides::new(new_strides);
            Some(self)
        } else {
            None
        }
    }

    fn is_contiguous(&self) -> bool {
        let res = self
            .strides
            .iter()
            .fold((true, usize::MAX), |(is_contiguous, last), stride| {
                if !is_contiguous || stride > last {
                    (false, stride)
                } else {
                    (true, stride)
                }
            });
        res.0
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    proptest! {
        // TODO: find a way to have arbitrary const generics?

        #[test]
        fn map_test(shape in Shape::arbitrary(), f in -1_f64..1.) {
            let map = TensorData::zeros(shape.clone()).map(|z| z + f);
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
}
