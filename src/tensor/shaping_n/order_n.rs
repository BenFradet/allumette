use std::collections::HashSet;

use super::const_iter::ConstIter;

#[derive(Debug)]
pub struct OrderN<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> OrderN<N> {
    pub fn new(data: [usize; N]) -> Option<Self> {
        let s = Self { data };
        if s.fits() {
            Some(s)
        } else {
            None
        }
    }

    pub fn range() -> Self {
        // this should be safe
        Self {
            data: (0..N).collect::<Vec<_>>().try_into().unwrap(),
        }
    }

    pub fn reverse(mut self) -> Self {
        self.data.reverse();
        self
    }

    pub fn iter(&self) -> ConstIter<N> {
        ConstIter::new(&self.data)
    }

    pub fn fits(&self) -> bool {
        let s1: HashSet<_> = self.data.into_iter().collect();
        let s2: HashSet<_> = (0..N).collect();
        s1 == s2
    }
}
