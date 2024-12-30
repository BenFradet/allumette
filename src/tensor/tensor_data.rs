use std::{collections::HashSet, ops::Index};

use proptest::{array, collection, prelude::*};

#[derive(Debug)]
pub struct TensorData<const N: usize> {
    data: Vec<f64>,
    shape: Shape<N>,
    strides: Strides<N>,
}

impl<const N: usize> TensorData<N> {
    fn new(data: Vec<f64>, shape: Shape<N>, strides: Strides<N>) -> Self {
        TensorData {
            data,
            shape,
            strides,
        }
    }

    fn position(&self, idx: Idx<N>) -> usize {
        idx.data
            .iter()
            .zip(self.strides.data.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    #[allow(clippy::needless_range_loop)]
    fn index(&self, pos: usize) -> Idx<N> {
        let mut res = [1; N];
        let mut mut_pos = pos;
        for i in 0..N {
            let s = self.strides.data[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Idx { data: res }
    }

    // TODO: look into use<'_, N>
    fn indices(&self) -> impl Iterator<Item = Idx<N>> + use<'_, N> {
        (0..self.size()).map(|i| self.index(i))
    }

    fn permute(mut self, order: Order<N>) -> Option<Self> {
        if order.fits_shape(&self.shape) {
            let mut new_shape = [0; N];
            let mut new_strides = [0; N];
            for (idx, value) in order.data.iter().enumerate() {
                new_shape[idx] = self.shape[*value];
                new_strides[idx] = self.strides[*value];
            }
            self.shape = Shape::new(new_shape);
            self.strides = Strides { data: new_strides };
            Some(self)
        } else {
            None
        }
    }

    fn is_contiguous(&self) -> bool {
        let res =
            self.strides
                .data
                .iter()
                .fold((true, usize::MAX), |(is_contiguous, last), stride| {
                    if !is_contiguous || *stride > last {
                        (false, *stride)
                    } else {
                        (true, *stride)
                    }
                });
        res.0
    }

    fn arbitrary() -> impl Strategy<Value = TensorData<N>> {
        Shape::arbitrary().prop_flat_map(|shape| {
            let size = shape.size;
            let data = collection::vec(0.0f64..1., size);
            (data, Just(shape))
        }).prop_map(|(data, shape)| {
            let strides: Strides<N> = (&shape).into();
            TensorData::new(data, shape, strides)
        })
    }
}

// all derives needed by the HashSet test
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Idx<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Idx<N> {
    fn iter(&self) -> IdxIter<N> {
        IdxIter {
            idx: self,
            index: 0,
        }
    }
}

struct IdxIter<'a, const N: usize> {
    idx: &'a Idx<N>,
    index: usize,
}

impl<'a, const N: usize> Iterator for IdxIter<'a, N> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.idx.data.len() {
            let res = self.idx.data[self.index];
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Strides<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Index<usize> for Strides<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> From<&Shape<N>> for Strides<N> {
    #[allow(clippy::needless_range_loop)]
    fn from(shape: &Shape<N>) -> Self {
        let mut res = [1; N];
        for i in (0..N - 1).rev() {
            res[i] = res[i + 1] * shape[i + 1];
        }
        Strides { data: res }
    }
}

// Clone needed by proptest's Just
#[derive(Clone, Debug, PartialEq)]
struct Shape<const N: usize> {
    data: [usize; N],
    size: usize,
}

impl<const N: usize> Shape<N> {
    fn new(data: [usize; N]) -> Self {
        let size = data.iter().fold(1, |acc, u| acc * u);
        Self {
            data,
            size,
        }
    }

    fn arbitrary() -> impl Strategy<Value = Shape<N>> {
        array::uniform(1usize..10).prop_map(|v| Shape::new(v))
    }
}

impl<const N: usize> Index<usize> for Shape<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

struct Order<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Order<N> {
    fn fits_shape(&self, shape: &Shape<N>) -> bool {
        let s1: HashSet<_> = self.data.into_iter().collect();
        let s2: HashSet<_> = (0..shape.size).collect();
        s1 == s2
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    proptest! {
        // TODO: find a way to have arbitrary const generics?
        #[test]
        fn position_test(tensor_data in TensorData::<4>::arbitrary()) {
            for idx in tensor_data.indices() {
                let pos = tensor_data.position(idx);
                assert!(pos < tensor_data.size());
            }
        }
        
        #[test]
        fn enumeration_test(tensor_data in TensorData::<4>::arbitrary()) {
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
        let idx1 = Idx { data: [1, 2] };
        let idx2 = Idx { data: [1, 2] };
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn stride_from_shape_test() -> () {
        let res: Strides<2> = (&Shape::new([5, 4])).into();
        assert_eq!([4, 1], res.data);
        let res2: Strides<3> = (&Shape::new([4, 2, 2])).into();
        assert_eq!([4, 2, 1], res2.data);
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([3, 5]);
        let strides = Strides { data: [5, 1] };
        let tensor = TensorData::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new([3, 5]), tensor.shape);
        assert_eq!(5, tensor.position(Idx { data: [1, 0] }));
        assert_eq!(7, tensor.position(Idx { data: [1, 2] }));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([5, 3]);
        let strides = Strides { data: [1, 5] };
        let tensor = TensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new([5, 3]), tensor.shape);
    }
}
