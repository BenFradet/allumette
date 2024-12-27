#[derive(Debug)]
pub struct TensorData<const N: usize> {
    data: Vec<f64>,
    shape: Shape<N>,
    strides: Strides<N>,
}

impl<const N: usize> TensorData<N> {
    fn new(data: Vec<f64>, shape: Shape<N>, strides: Strides<N>) -> Self {
        TensorData {
            data,
            shape,
            strides,
        }
    }

    fn position(&self, idx: Index<N>) -> usize {
        idx.data
            .iter()
            .zip(self.strides.data.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
    }

    #[allow(clippy::needless_range_loop)]
    fn index(&self, pos: usize) -> Index<N> {
        let mut res = [1; N];
        let mut mut_pos = pos;
        for i in 0..N {
            let s = self.strides.data[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Index { data: res }
    }

    fn is_contiguous(&self) -> bool {
        let res =
            self.strides
                .data
                .iter()
                .fold((true, usize::MAX), |(is_contiguous, last), stride| {
                    if !is_contiguous || *stride > last {
                        (false, *stride)
                    } else {
                        (true, *stride)
                    }
                });
        res.0
    }
}

#[derive(Debug)]
struct Index<const N: usize> {
    data: [usize; N],
}

#[derive(Debug)]
struct Order<const N: usize> {
    data: [usize; N],
}

#[derive(Debug)]
struct Strides<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> From<&Shape<N>> for Strides<N> {
    #[allow(clippy::needless_range_loop)]
    fn from(shape: &Shape<N>) -> Self {
        let mut res = [1; N];
        for i in (0..N - 1).rev() {
            res[i] = res[i + 1] * shape.data[i + 1];
        }
        Strides { data: res }
    }
}

#[derive(Debug, PartialEq)]
struct Shape<const N: usize> {
    data: [usize; N],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_from_shape_test() -> () {
        let res: Strides<2> = (&Shape { data: [5, 4] }).into();
        assert_eq!([4, 1], res.data);
        let res2: Strides<3> = (&Shape { data: [4, 2, 2] }).into();
        assert_eq!([4, 2, 1], res2.data);
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape { data: [3, 5] };
        let strides = Strides { data: [5, 1] };
        let tensor = TensorData::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape { data: [3, 5] }, tensor.shape);
        assert_eq!(5, tensor.position(Index { data: [1, 0] }));
        assert_eq!(7, tensor.position(Index { data: [1, 2] }));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape { data: [5, 3] };
        let strides = Strides { data: [1, 5] };
        let tensor = TensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape { data: [5, 3] }, tensor.shape);
    }
}
