use std::collections::HashSet;

use super::iter::Iter;

#[derive(Debug)]
pub struct Order {
    data: Vec<usize>,
}

impl Order {
    pub fn new(data: Vec<usize>) -> Option<Self> {
        let len = data.len();
        let s = Self { data };
        if s.fits(len) {
            Some(s)
        } else {
            None
        }
    }

    pub fn range(n: usize) -> Self {
        // this should be safe
        Self {
            data: (0..n).collect::<Vec<_>>().try_into().unwrap(),
        }
    }

    pub fn reverse(mut self) -> Self {
        self.data.reverse();
        self
    }

    pub fn iter(&self) -> Iter {
        Iter::new(&self.data)
    }

    // TODO: refactor
    pub fn fits(&self, n: usize) -> bool {
        let s1: HashSet<_> = self.data.iter().copied().collect();
        let s2: HashSet<_> = (0..n).collect();
        s1 == s2
    }
}
