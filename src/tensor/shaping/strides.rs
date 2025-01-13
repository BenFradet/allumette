use std::ops::Index;

use crate::util::const_iter::ConstIter;

use super::{idx::Idx, shape::Shape};

#[derive(Clone, Debug)]
pub struct Strides<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> Strides<N> {
    pub fn new(data: [usize; N]) -> Self {
        Self { data }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn idx(&self, pos: usize) -> Idx<N> {
        let mut res = [1; N];
        let mut mut_pos = pos;
        for i in 0..N {
            let s = self[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Idx::new(res)
    }

    pub fn position(&self, idx: &Idx<N>) -> usize {
        idx.iter()
            .zip(self.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
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
    use proptest::proptest;

    use crate::tensor::tensor::Tensor;

    use super::*;

    #[test]
    fn stride_from_shape_test() -> () {
        let res: Strides<2> = (&Shape::new([5, 4])).into();
        assert_eq!([4, 1], res.data);
        let res2: Strides<3> = (&Shape::new([4, 2, 2])).into();
        assert_eq!([4, 2, 1], res2.data);
    }

    proptest! {
        #[test]
        fn position_test(tensor_data in Tensor::<4>::arbitrary()) {
            for idx in tensor_data.indices() {
                let pos = tensor_data.strides.position(&idx);
                assert!(pos < tensor_data.size());
            }
        }
    }
}
