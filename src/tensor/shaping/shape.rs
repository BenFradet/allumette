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
    // https://hackmd.io/OZG_XiLFRs2Xmw5s39jRzA
    // "[(); Self::max(M, N)]:" const well-formed bound
    pub fn broadcast<const M: usize>(self, other: Shape<M>) -> Option<Shape<{ Self::max(M, N) }>>
    where
        [(); Self::max(M, N)]:,
    {
        let padded_n = self.pad_left::<{ Self::max(M, N) }>(1);
        let padded_m = other.pad_left::<{ Self::max(M, N) }>(1);
        let mut res = [0; Self::max(M, N)];
        let mut flag = false;

        for i in 0..Self::max(M, N) {
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

    pub fn pad_left<const M: usize>(self, cnst: usize) -> Shape<M> {
        // TODO: can't do if N == M self because compiler doesn't know that
        let mut res = [cnst; M];
        if N < M {
            let offset = M - N;
            for (i, item) in res.iter_mut().enumerate().take(M).skip(offset) {
                let ni = i - offset;
                *item = self.data[ni];
            }
        } else {
            res[..M].copy_from_slice(&self.data[..M]);
        }
        Shape::new(res)
    }

    pub const fn max(x: usize, y: usize) -> usize {
        if x < y {
            y
        } else {
            x
        }
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
    fn broadcast_test() -> () {
        let s1 = Shape::new([2, 3, 1]);
        let s2 = Shape::new([7, 2, 3, 5]);
        assert_eq!(Some(Shape::new([7, 2, 3, 5])), s1.broadcast(s2));

        let s1 = Shape::new([1]);
        let s2 = Shape::new([5, 5]);
        assert_eq!(Some(Shape::new([5, 5])), s1.broadcast(s2));

        let s1 = Shape::new([5, 5]);
        let s2 = Shape::new([1]);
        assert_eq!(Some(Shape::new([5, 5])), s1.broadcast(s2));

        let s1 = Shape::new([1, 5, 5]);
        let s2 = Shape::new([5, 5]);
        assert_eq!(Some(Shape::new([1, 5, 5])), s1.broadcast(s2));

        let s1 = Shape::new([5, 1, 5, 1]);
        let s2 = Shape::new([1, 5, 1, 5]);
        assert_eq!(Some(Shape::new([5, 5, 5, 5])), s1.broadcast(s2));

        let s1 = Shape::new([5, 7, 5, 1]);
        let s2 = Shape::new([1, 5, 1, 5]);
        assert_eq!(None, s1.broadcast(s2));

        let s1 = Shape::new([5, 2]);
        let s2 = Shape::new([5]);
        assert_eq!(None, s1.broadcast(s2));

        let s1 = Shape::new([2, 5]);
        let s2 = Shape::new([5]);
        assert_eq!(Some(Shape::new([2, 5])), s1.broadcast(s2));
    }

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
