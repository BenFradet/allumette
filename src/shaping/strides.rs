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

    pub fn data(&self) -> &[usize] {
        &self.data
    }

    #[inline(always)]
    pub fn idx(&self, mut pos: usize) -> Idx {
        let res = self.data.iter().map(|&s| {
            let idx = pos / s;
            pos %= s;
            idx
        }).collect();
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
    fn stride_from_shape_test() {
        let res: Strides = (&Shape::new(vec![5, 4])).into();
        assert_eq!(vec![4, 1], res.data);
        let res2: Strides = (&Shape::new(vec![4, 2, 2])).into();
        assert_eq!(vec![4, 2, 1], res2.data);
    }

    #[ignore = "dbg"]
    #[test]
    fn slide_test() {
        let shape = Shape::new(vec![3, 3]);
        println!("shape {shape:?}");
        let strides: Strides = (&shape).into();
        println!("strides {strides:?}");
        let shape_reduced = Shape::new(vec![3, 1]);
        println!("shape reduced {shape_reduced:?}");
        let strides_reduced: Strides = (&shape_reduced).into();
        println!("strides_reduced {strides_reduced:?}");
        let dim = 1;
        let mut idx = strides_reduced.idx(2);
        let out_pos = strides_reduced.position(&idx);
        println!("out pos {out_pos:?}\n");
        println!("idx {idx:?}\n");
        for j in 0..3 {
            println!("j {j:?}");
            idx[dim] = j;
            println!("idx {idx:?}");
            let pos = strides.position(&idx);
            println!("pos {pos:?}");
        }
        assert!(false)
    }

    // TODO: reintroduce
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
