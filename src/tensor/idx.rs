use crate::util::const_iter::ConstIter;

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
}
