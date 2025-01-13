use proptest::{array, prelude::Strategy};

use crate::util::const_iter::ConstIter;

use super::shape::Shape;

// all derives needed by the HashSet test
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Idx<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Idx<N> {
    pub fn new(data: [usize; N]) -> Self {
        Self { data }
    }

    pub fn iter(&self) -> ConstIter<N> {
        ConstIter::new(&self.data)
    }

    // reduce self into smaller index
    pub fn broadcast<const M: usize>(&self, reference_shape: &Shape<M>) -> Option<Idx<M>> {
        if N >= M {
            let offset = N - M;
            let mut res = [0; M];
            for i in 0..M {
                if reference_shape[i] != 1 {
                    res[i] = self.data[offset + i]
                }
            }
            Some(Idx::new(res))
        } else {
            None
        }
    }

    pub fn reverse(mut self) -> Self {
        self.data.reverse();
        self
    }

    pub fn arbitrary() -> impl Strategy<Value = Idx<N>> {
        Shape::arbitrary()
            .prop_flat_map(|shape: Shape<N>| array::uniform(0usize..shape.size))
            .prop_map(Idx::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() -> () {
        let s = Shape::new([2, 3]);
        let i = Idx::new([0, 0, 1]);
        assert_eq!(Some(Idx::new([0, 1])), i.broadcast(&s));

        let s = Shape::new([2, 3, 3]);
        let i = Idx::new([0, 0, 1]);
        assert_eq!(Some(Idx::new([0, 0, 1])), i.broadcast(&s));

        let s = Shape::new([2, 3, 3, 4]);
        let i = Idx::new([0, 0, 1]);
        assert_eq!(None, i.broadcast(&s));
    }
}
