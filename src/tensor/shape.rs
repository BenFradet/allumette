use std::ops::Index;

use proptest::{array, prelude::Strategy};

// Clone needed by proptest's Just
#[derive(Clone, Debug, PartialEq)]
pub struct Shape<const N: usize> {
    data: [usize; N],
    pub size: usize,
}

impl<const N: usize> Shape<N> {
    pub fn new(data: [usize; N]) -> Self {
        let size = data.iter().product::<usize>();
        Self { data, size }
    }

    // feature(generic_const_exprs)
    // https://github.com/rust-lang/rust/issues/76560
    // https://users.rust-lang.org/t/operations-on-const-generic-parameters-as-a-generic-parameter/78865/2
    pub fn broadcast<const M: usize>(&self, other: &Shape<M>) -> Shape<{Self::max(M, N)}>
    where [(); Self::max(M, N)]: {
        let res = [0; Self::max(M, N)];
        // N != M => padLeft with 1s
        // ∀ y, y = f(1)
        // ∀ x != 1, ¬∃ y = f(x)
        Shape::new(res)
    }

    pub fn pad_left<const M: usize>(self, cnst: usize) -> Shape<M> {
        let mut res = [cnst; M];
        if N < M {
            let offset = M - N;
            for i in offset..M {
                let ni = i - offset;
                res[i] = self.data[ni];
            }
        } else {
            for i in 0..M {
                res[i] = self.data[i];
            }
        }
        Shape::new(res)
    }

    pub const fn max(x: usize, y: usize) -> usize {
        if x < y { y } else { x }
    }

    pub fn arbitrary() -> impl Strategy<Value = Shape<N>> {
        array::uniform(1usize..3).prop_map(Shape::new)
    }
}

impl<const N: usize> Index<usize> for Shape<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn pad_left_test() -> () {
        let s = Shape::new([1, 2, 1, 2]);
        let pad = s.pad_left::<6>(0);
        assert_eq!([0, 0, 1, 2, 1, 2], pad.data);

        let s = Shape::new([]);
        let pad = s.pad_left::<6>(0);
        assert_eq!([0, 0, 0, 0, 0, 0], pad.data);

        let s = Shape::new([1, 2, 1, 2]);
        let pad = s.pad_left::<0>(0);
        assert_eq!([0; 0], pad.data);

        let s = Shape::new([1, 2, 3, 4]);
        let pad = s.pad_left::<2>(0);
        assert_eq!([1, 2], pad.data);
    }
}
