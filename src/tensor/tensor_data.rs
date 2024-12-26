// TODO: stride and shape part of tensor data?
#[derive(Debug)]
pub struct TensorData<const N: usize> {
    data: Vec<f64>,
    stride: Stride<N>,
    shape: Shape<N>,
}

impl<const N: usize> TensorData<N> {
    fn position(&self, idx: Index<N>) -> usize {
        idx.data
            .iter()
            .zip(self.stride.data.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
    }

    fn index(&self, pos: usize) -> Index<N> {
        let stride = Self::stride_from_shape(&self.shape);
        let mut res = [1; N];
        let mut mut_pos = pos;
        for i in 0..N {
            let s = stride.data[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Index { data: res }
    }

    // TODO: impl From
    fn stride_from_shape(shape: &Shape<N>) -> Stride<N> {
        let mut res = [1; N];
        let mut offset = 1;
        for i in 0..N - 1 {
            let idx = N - i - 1;
            let s = shape.data[idx];
            res[i] = s * offset;
            offset = shape.data[idx] * offset;
        }
        Stride { data: res }
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

#[derive(Debug)]
struct Shape<const N: usize> {
    data: [usize; N],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_from_shape_test() -> () {
        let res = TensorData::stride_from_shape(&Shape { data: [5, 4] });
        assert_eq!([4, 1], res.data);
    }
}
