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
        let shape_a = Shape::new(vec![1, 2]);
        let strides_a: Strides = (&shape_a).into();
        println!("strides a {strides_a:?}");
        let shape_b = Shape::new(vec![2, 1]);
        let strides_b: Strides = (&shape_b).into();
        println!("strides b {strides_b:?}");
        let bc = shape_a.broadcast(&shape_b).unwrap();
        println!("{bc:?}");
        let strides: Strides = (&bc).into();
        println!("{strides:?}");
        let idx = strides.idx(3);
        println!("idx {idx:?}");
        let idxa = idx.broadcast(&shape_a).unwrap();
        println!("idxa {idxa:?}");
        let idxb = idx.broadcast(&shape_b).unwrap();
        println!("idxb {idxb:?}");
        let posa = strides_a.position(&idxa);
        println!("posa {posa:?}");
        let posb = strides_b.position(&idxb);
        println!("posb {posb:?}");
        let pos = strides.position(&idx);
        println!("pos {pos:?}");
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
