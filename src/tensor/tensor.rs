use std::sync::Arc;

use proptest::{collection, prelude::*};

use crate::util::max::max;

use super::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

#[derive(Debug)]
pub struct Tensor<const N: usize> {
    pub data: Arc<Vec<f64>>,
    pub shape: Shape<N>,
    pub strides: Strides<N>,
}

impl<const N: usize> Tensor<N> {
    fn new(data: Vec<f64>, shape: Shape<N>, strides: Strides<N>) -> Self {
        Tensor {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    // TODO: we could do without any broadcasting as this is an endomorphism
    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let len = self.data.len();
        let mut out = Vec::with_capacity(len);
        // TODO: add an iterator
        for i in 0..len {
            let idx = self.strides.idx(i);
            let pos = self.strides.position(&idx);
            let value = self.data[pos];
            let to_idx = idx.broadcast(&self.shape).unwrap();
            let to_pos = self.strides.position(&to_idx);
            out[to_pos] = f(value);
        }
        // cloning of stack-allocated arrays should be cheap
        Tensor::new(out, self.shape.clone(), self.strides.clone())
    }

    // feature(generic_const_exprs)
    fn zip<const M: usize>(&self, other: Tensor<M>, f: impl Fn(f64, f64) -> f64) -> Option<Tensor<{ max(M, N) }>>
    where
        [(); max(M, N)]:,
    {
        let shape = self.shape.broadcast(&other.shape)?;
        let strides: Strides<_> = (&shape).into();
        let len = shape.size;
        let mut out = Vec::with_capacity(len);
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
        Some(Tensor::new(out, shape, strides))
    }

    pub fn size(&self) -> usize {
        self.shape.size
    }

    // TODO: look into use<'_, N>
    pub fn indices(&self) -> impl Iterator<Item = Idx<N>> + use<'_, N> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    fn permute(mut self, order: &Order<N>) -> Option<Self> {
        if order.fits() {
            let mut new_shape = [0; N];
            let mut new_strides = [0; N];
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

    pub fn arbitrary() -> impl Strategy<Value = Tensor<N>> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f64..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: Strides<N> = (&shape).into();
                Tensor::new(data, shape, strides)
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
        fn permute_test(tensor_data in Tensor::<4>::arbitrary(), idx in Idx::<4>::arbitrary()) {
            let reversed_index = idx.clone().reverse();
            let pos = tensor_data.strides.position(&idx);
            let order = Order::range().reverse();
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
        fn enumeration_test(tensor_data in Tensor::<4>::arbitrary()) {
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
        let idx1 = Idx::new([1, 2]);
        let idx2 = Idx::new([1, 2]);
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([3, 5]);
        let strides = Strides::new([5, 1]);
        let tensor = Tensor::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new([3, 5]), tensor.shape);
        assert_eq!(5, tensor.strides.position(&Idx::new([1, 0])));
        assert_eq!(7, tensor.strides.position(&Idx::new([1, 2])));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([5, 3]);
        let strides = Strides::new([1, 5]);
        let tensor = Tensor::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new([5, 3]), tensor.shape);
    }
}
