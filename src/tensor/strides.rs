use std::ops::Index;

use crate::util::const_iter::ConstIter;

use super::shape::Shape;

#[derive(Debug)]
pub struct Strides<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Strides<N> {
    pub fn new(data: [usize; N]) -> Self {
        Self { data }
    }

    pub fn iter(&self) -> ConstIter<N> {
        ConstIter::new(&self.data)
    }
}

impl<const N: usize> Index<usize> for Strides<N> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> From<&Shape<N>> for Strides<N> {
    // TODO: fixme
    #[allow(clippy::needless_range_loop)]
    fn from(shape: &Shape<N>) -> Self {
        let mut res = [1; N];
        for i in (0..N - 1).rev() {
            res[i] = res[i + 1] * shape[i + 1];
        }
        Strides { data: res }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_from_shape_test() -> () {
        let res: Strides<2> = (&Shape::new([5, 4])).into();
        assert_eq!([4, 1], res.data);
        let res2: Strides<3> = (&Shape::new([4, 2, 2])).into();
        assert_eq!([4, 2, 1], res2.data);
    }
}
