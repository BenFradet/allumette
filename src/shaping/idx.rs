use std::ops::{Index, IndexMut};

use proptest::prelude::Strategy;

use super::{iter::Iter, shape::Shape};

// all derives needed by the HashSet test
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Idx {
    data: Vec<usize>,
}

impl Idx {
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }

    pub fn iter(&self) -> Iter {
        Iter::new(&self.data)
    }

    // reduce self into smaller index
    pub fn broadcast(&self, reference_shape: &Shape) -> Option<Idx> {
        let n = self.data.len();
        let m = reference_shape.data().len();
        if n >= m {
            let offset = n - m;
            let mut res = vec![0; m];
            for i in 0..m {
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

    pub fn arbitrary() -> impl Strategy<Value = Idx> {
        Shape::arbitrary()
            .prop_flat_map(|shape: Shape| {
                proptest::collection::vec(0usize..shape.size, shape.data().len())
            })
            .prop_map(Idx::new)
    }
}

impl Index<usize> for Idx {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Idx {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() -> () {
        let s = Shape::new(vec![2, 3]);
        let i = Idx::new(vec![0, 0, 1]);
        assert_eq!(Some(Idx::new(vec![0, 1])), i.broadcast(&s));

        let s = Shape::new(vec![2, 3, 3]);
        let i = Idx::new(vec![0, 0, 1]);
        assert_eq!(Some(Idx::new(vec![0, 0, 1])), i.broadcast(&s));

        let s = Shape::new(vec![2, 3, 3, 4]);
        let i = Idx::new(vec![0, 0, 1]);
        assert_eq!(None, i.broadcast(&s));
    }
}
