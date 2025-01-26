use std::ops::{Index, IndexMut};

use proptest::{array, prelude::Strategy};

use super::{const_iter::ConstIter, shape_n::ShapeN};

// all derives needed by the HashSet test
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IdxN<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> IdxN<N> {
    pub fn new(data: [usize; N]) -> Self {
        Self { data }
    }

    pub fn iter(&self) -> ConstIter<N> {
        ConstIter::new(&self.data)
    }

    // reduce self into smaller index
    pub fn broadcast<const M: usize>(&self, reference_shape: &ShapeN<M>) -> Option<IdxN<M>> {
        if N >= M {
            let offset = N - M;
            let mut res = [0; M];
            for i in 0..M {
                if reference_shape[i] != 1 {
                    res[i] = self.data[offset + i]
                }
            }
            Some(IdxN::new(res))
        } else {
            None
        }
    }

    pub fn reverse(mut self) -> Self {
        self.data.reverse();
        self
    }

    pub fn arbitrary() -> impl Strategy<Value = IdxN<N>> {
        ShapeN::arbitrary()
            .prop_flat_map(|shape: ShapeN<N>| array::uniform(0usize..shape.size))
            .prop_map(IdxN::new)
    }
}

impl<const N: usize> IndexMut<usize> for IdxN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize> Index<usize> for IdxN<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() -> () {
        let s = ShapeN::new([2, 3]);
        let i = IdxN::new([0, 0, 1]);
        assert_eq!(Some(IdxN::new([0, 1])), i.broadcast(&s));

        let s = ShapeN::new([2, 3, 3]);
        let i = IdxN::new([0, 0, 1]);
        assert_eq!(Some(IdxN::new([0, 0, 1])), i.broadcast(&s));

        let s = ShapeN::new([2, 3, 3, 4]);
        let i = IdxN::new([0, 0, 1]);
        assert_eq!(None, i.broadcast(&s));
    }
}
