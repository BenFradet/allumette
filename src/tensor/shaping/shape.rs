use std::ops::Index;

use proptest::prelude::Strategy;

use crate::util::max::max;

#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    data: Vec<usize>,
    pub size: usize,
}

impl Shape {
    pub fn new(data: Vec<usize>) -> Self {
        let size = data.iter().product::<usize>();
        Self { data, size }
    }

    pub fn data(&self) -> &[usize] {
        &self.data
    }

    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let n = self.data.len();
        let m = other.data.len();
        let max = max(m, n);
        let padded_n = self.pad_left(max, 1);
        let padded_m = other.pad_left(max, 1);
        let mut res = vec![0; max];
        let mut flag = false;

        for i in 0..max {
            let n = padded_n[i];
            let m = padded_m[i];
            if n == 1 {
                // ∀ x, f(1, x) = x
                res[i] = m;
            } else if m == 1 {
                // ∀ x, f(x, 1) = x
                res[i] = n;
            } else if m == n {
                // ∀ x, f(x, x) = x
                res[i] = n;
            } else {
                // ∀ x != 1, ¬∃ y != x, f(x, y) = z
                flag = true;
                break;
            }
        }

        if flag {
            None
        } else {
            Some(Shape::new(res))
        }
    }

    pub fn pad_left(&self, m: usize, cnst: usize) -> Shape {
        let n = self.data.len();
        let mut res = vec![cnst; m];
        println!("{:?}", res);
        if n < m {
            let offset = m - n;
            for (i, item) in res.iter_mut().enumerate().take(m).skip(offset) {
                let ni = i - offset;
                *item = self.data[ni];
            }
        } else {
            res[..m].copy_from_slice(&self.data[..m]);
        }
        Shape::new(res)
    }

    pub fn arbitrary() -> impl Strategy<Value = Shape> {
        proptest::collection::vec(1_usize..3, 4).prop_map(Shape::new)
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() -> () {
        let s1 = Shape::new(vec![2, 3, 1]);
        let s2 = Shape::new(vec![7, 2, 3, 5]);
        assert_eq!(Some(Shape::new(vec![7, 2, 3, 5])), s1.broadcast(&s2));

        let s1 = Shape::new(vec![1]);
        let s2 = Shape::new(vec![5, 5]);
        assert_eq!(Some(Shape::new(vec![5, 5])), s1.broadcast(&s2));

        let s1 = Shape::new(vec![5, 5]);
        let s2 = Shape::new(vec![1]);
        assert_eq!(Some(Shape::new(vec![5, 5])), s1.broadcast(&s2));

        let s1 = Shape::new(vec![1, 5, 5]);
        let s2 = Shape::new(vec![5, 5]);
        assert_eq!(Some(Shape::new(vec![1, 5, 5])), s1.broadcast(&s2));

        let s1 = Shape::new(vec![5, 1, 5, 1]);
        let s2 = Shape::new(vec![1, 5, 1, 5]);
        assert_eq!(Some(Shape::new(vec![5, 5, 5, 5])), s1.broadcast(&s2));

        let s1 = Shape::new(vec![5, 7, 5, 1]);
        let s2 = Shape::new(vec![1, 5, 1, 5]);
        assert_eq!(None, s1.broadcast(&s2));

        let s1 = Shape::new(vec![5, 2]);
        let s2 = Shape::new(vec![5]);
        assert_eq!(None, s1.broadcast(&s2));

        let s1 = Shape::new(vec![2, 5]);
        let s2 = Shape::new(vec![5]);
        assert_eq!(Some(Shape::new(vec![2, 5])), s1.broadcast(&s2));
    }

    #[test]
    fn pad_left_test() -> () {
        let s = Shape::new(vec![1, 2, 1, 2]);
        let pad = s.pad_left(6, 0);
        assert_eq!(vec![0, 0, 1, 2, 1, 2], pad.data);

        let s = Shape::new(vec![]);
        let pad = s.pad_left(6, 0);
        assert_eq!(vec![0, 0, 0, 0, 0, 0], pad.data);

        let s = Shape::new(vec![1, 2, 1, 2]);
        let pad = s.pad_left(0, 0);
        assert_eq!(vec![0; 0], pad.data);

        let s = Shape::new(vec![1, 2, 3, 4]);
        let pad = s.pad_left(2, 0);
        assert_eq!(vec![1, 2], pad.data);
    }
}
