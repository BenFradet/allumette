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
