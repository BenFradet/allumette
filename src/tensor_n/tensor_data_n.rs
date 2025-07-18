use std::sync::Arc;

use proptest::{collection, prelude::*};

use crate::util::{
    max::max,
    type_if::{TypeIf, TypeTrue},
};

use super::shaping_n::{idx_n::IdxN, order_n::OrderN, shape_n::ShapeN, strides_n::StridesN};

#[derive(Clone, Debug)]
pub struct TensorDataN<const N: usize> {
    pub data: Arc<Vec<f32>>,
    pub shape: ShapeN<N>,
    pub strides: StridesN<N>,
}

impl<const N: usize> TensorDataN<N> {
    fn new(data: Vec<f32>, shape: ShapeN<N>, strides: StridesN<N>) -> Self {
        TensorDataN {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn zeros(shape: ShapeN<N>) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        TensorDataN {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn ones(shape: ShapeN<N>) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        TensorDataN {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn map(mut self, f: impl Fn(f32) -> f32) -> Self {
        let len = self.size();
        let mut out = vec![0.; len];
        // TODO: add an iterator
        for (i, d) in self.data.iter().enumerate() {
            out[i] = f(*d);
        }
        self.data = Arc::new(out);
        self
    }

    fn map_(mut self, f: impl Fn(f32) -> f32) -> Self {
        let len = self.size();
        let mut out = vec![0.; len];
        for i in 0..len {
            let idx = self.strides.idx(i);
            let pos = self.strides.position(&idx);
            let value = self.data[pos];
            let to_idx = idx.broadcast(&self.shape).unwrap();
            let to_pos = self.strides.position(&to_idx);
            out[to_pos] = f(value);
        }
        self.data = Arc::new(out);
        self
    }

    // feature(generic_const_exprs)
    pub fn zip<const M: usize>(
        &self,
        other: &TensorDataN<M>,
        f: impl Fn(f32, f32) -> f32,
    ) -> Option<TensorDataN<{ max(M, N) }>>
    where
        [(); max(M, N)]:,
    {
        let shape = self.shape.broadcast(&other.shape)?;
        let strides: StridesN<_> = (&shape).into();
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
        Some(TensorDataN::new(out, shape, strides))
    }

    pub fn zip_n(mut self, other: &TensorDataN<N>, f: impl Fn(f32, f32) -> f32) -> TensorDataN<N> {
        let len = self.shape.size;
        let mut out = vec![0.; len];
        for (i, o) in out.iter_mut().enumerate() {
            *o = f(self.data[i], other.data[i]);
        }
        self.data = Arc::new(out);
        self
    }

    pub fn reduce<const M: usize>(&self, f: impl Fn(f32, f32) -> f32) -> TensorDataN<N>
    where
        TypeIf<{ M < N }>: TypeTrue,
    {
        let mut shape_data = *self.shape.data();
        shape_data[M] = 1;
        let shape = ShapeN::new(shape_data);
        let strides: StridesN<_> = (&shape).into();
        let len = shape.size;
        let mut out = vec![0.; len];
        for i in 0..len {
            let out_idx = strides.idx(i);
            let out_pos = strides.position(&out_idx);
            for j in 0..self.shape[M] {
                let mut self_idx = out_idx.clone();
                self_idx[M] = j;
                let self_pos = self.strides.position(&self_idx);
                let v_out = out[out_pos];
                let v_self = self.data[self_pos];
                out[out_pos] = f(v_out, v_self);
            }
        }
        TensorDataN::new(out, shape, strides)
    }

    pub fn size(&self) -> usize {
        self.shape.size
    }

    // TODO: look into use<'_, N>
    pub fn indices(&self) -> impl Iterator<Item = IdxN<N>> + use<'_, N> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    fn permute(mut self, order: &OrderN<N>) -> Option<Self> {
        if order.fits() {
            let mut new_shape = [0; N];
            let mut new_strides = [0; N];
            for (idx, value) in order.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                new_strides[idx] = self.strides[value];
            }
            self.shape = ShapeN::new(new_shape);
            self.strides = StridesN::new(new_strides);
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

    pub fn arbitrary() -> impl Strategy<Value = TensorDataN<N>> {
        ShapeN::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f32..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: StridesN<N> = (&shape).into();
                TensorDataN::new(data, shape, strides)
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
        fn map_test(shape in ShapeN::<4>::arbitrary(), f in -1_f32..1.) {
            let map_ = TensorDataN::zeros(shape.clone()).map_(|z| z + f);
            assert_eq!(shape.size, map_.data.len());
            assert!(map_.data.iter().all(|e| *e == f));
            let map = TensorDataN::zeros(shape.clone()).map(|z| z + f);
            assert_eq!(shape.size, map.data.len());
            assert!(map.data.iter().all(|e| *e == f));
        }


        #[test]
        fn zeros_test(shape in ShapeN::<4>::arbitrary()) {
            let zeros = TensorDataN::zeros(shape.clone());
            assert_eq!(shape.size, zeros.data.len());
            assert!(zeros.data.iter().all(|f| *f == 0.));
        }

        #[test]
        fn permute_test(tensor_data in TensorDataN::<4>::arbitrary(), idx in IdxN::<4>::arbitrary()) {
            let reversed_index = idx.clone().reverse();
            let pos = tensor_data.strides.position(&idx);
            let order = OrderN::range().reverse();
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
        fn enumeration_test(tensor_data in TensorDataN::<4>::arbitrary()) {
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
        let idx1 = IdxN::new([1, 2]);
        let idx2 = IdxN::new([1, 2]);
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = ShapeN::new([3, 5]);
        let strides = StridesN::new([5, 1]);
        let tensor = TensorDataN::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(ShapeN::new([3, 5]), tensor.shape);
        assert_eq!(5, tensor.strides.position(&IdxN::new([1, 0])));
        assert_eq!(7, tensor.strides.position(&IdxN::new([1, 2])));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = ShapeN::new([5, 3]);
        let strides = StridesN::new([1, 5]);
        let tensor = TensorDataN::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(ShapeN::new([5, 3]), tensor.shape);
    }
}
