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

    pub fn arbitrary() -> impl Strategy<Value = Shape<N>> {
        array::uniform(1usize..10).prop_map(Shape::new)
    }
}

impl<const N: usize> Index<usize> for Shape<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
