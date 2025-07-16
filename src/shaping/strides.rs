use std::ops::Index;

use super::{idx::Idx, iter::Iter, shape::Shape};

#[derive(Clone, Debug, PartialEq)]
pub struct Strides {
    data: Vec<usize>,
}

impl Strides {
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }

    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    pub fn idx(&self, pos: usize) -> Idx {
        let n = self.data.len();
        let mut res = vec![1; n];
        let mut mut_pos = pos;
        for i in 0..n {
            let s = self[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Idx::new(res)
    }

    #[inline(always)]
    pub fn position(&self, idx: &Idx) -> usize {
        idx.iter()
            .zip(self.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter::new(&self.data)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Index<usize> for Strides {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl From<&Shape> for Strides {
    #[allow(clippy::needless_range_loop)]
    fn from(shape: &Shape) -> Self {
        let n = shape.data().len();
        let mut res = vec![1; n];
        for i in (0..n - 1).rev() {
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
        let res: Strides = (&Shape::new(vec![5, 4])).into();
        assert_eq!(vec![4, 1], res.data);
        let res2: Strides = (&Shape::new(vec![4, 2, 2])).into();
        assert_eq!(vec![4, 2, 1], res2.data);
    }

    //proptest! {
    //    #[test]
    //    fn position_test(tensor_data in Tensor::<4>::arbitrary()) {
    //        for idx in tensor_data.indices() {
    //            let pos = tensor_data.strides.position(&idx);
    //            assert!(pos < tensor_data.size());
    //        }
    //    }
    //}
}
