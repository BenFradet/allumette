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
