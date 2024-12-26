#[derive(Debug)]
pub struct TensorData<const N: usize> {
    data: Vec<f64>,
    strides: Stride<N>,
    shape: Shape<N>,
}

impl<const N: usize> TensorData<N> {
    fn new(data: Vec<f64>, strides: Stride<N>, shape: Shape<N>) -> Self {
        TensorData {
            data,
            strides,
            shape,
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
        let res = self.strides.data.iter().fold((true, usize::MAX), |(is_contiguous, last), stride| {
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
struct Stride<const N: usize> {
    data: [usize; N],
}

impl<const N: usize> From<&Shape<N>> for Stride<N> {
    #[allow(clippy::needless_range_loop)]
    fn from(shape: &Shape<N>) -> Self {
        let mut res = [1; N];
        let mut offset = 1;
        for i in 0..N - 1 {
            let idx = N - i - 1;
            let s = shape.data[idx];
            res[i] = s * offset;
            offset *= shape.data[idx];
        }
        Stride { data: res }
    }
}

#[derive(Debug)]
struct Shape<const N: usize> {
    data: [usize; N],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_from_shape_test() -> () {
        let res: Stride<2> = (&Shape { data: [5, 4] }).into();
        assert_eq!([4, 1], res.data);
    }
}
