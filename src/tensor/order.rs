use std::collections::HashSet;

use crate::util::const_iter::ConstIter;

use super::shape::Shape;

pub struct Order<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Order<N> {
    pub fn new(data: [usize; N]) -> Self {
        Self { data }
    }

    pub fn iter(&self) -> ConstIter<N> {
        ConstIter::new(&self.data)
    }

    pub fn fits_shape(&self, shape: &Shape<N>) -> bool {
        let s1: HashSet<_> = self.data.into_iter().collect();
        let s2: HashSet<_> = (0..shape.size).collect();
        s1 == s2
    }
}
