use std::{ops::Index, sync::Arc};

use proptest::{collection, prelude::*};

use crate::shaping::{idx::Idx, shape::Shape, shaped::Shaped, strides::Strides};

#[derive(Clone, Debug)]
pub struct TensorData {
    pub data: Arc<Vec<f64>>,
    pub shape: Shape,
    pub strides: Strides,
}

impl TensorData {
    pub fn new(data: Vec<f64>, shape: Shape, strides: Strides) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn epsilon(shape: Shape, idx: &Idx, eps: f64) -> Self {
        let strides: Strides = (&shape).into();
        let mut data = vec![0.; shape.size];
        data[strides.position(idx)] = eps;
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    pub fn indices(&self) -> impl Iterator<Item = Idx> + use<'_> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    pub fn arbitrary() -> impl Strategy<Value = TensorData> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f64..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: Strides = (&shape).into();
                TensorData::new(data, shape, strides)
            })
    }
}

impl Index<Idx> for TensorData {
    type Output = f64;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.strides.position(&index)]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn assert_tensor_eq(t1: &TensorData, t2: &TensorData) -> () {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.data, t2.data);
    }

    proptest! {
        #[test]
        fn zeros_test(shape in Shape::arbitrary()) {
            let zeros = TensorData::zeros(shape.clone());
            assert_eq!(shape.size, zeros.data.len());
            assert!(zeros.data.iter().all(|f| *f == 0.));
        }

        #[test]
        fn enumeration_test(tensor_data in TensorData::arbitrary()) {
            let indices: Vec<_> = tensor_data.indices().collect();
            let count = indices.len();
            assert_eq!(tensor_data.size(), count);
            let set: HashSet<_> = indices.clone().into_iter().collect();
            assert_eq!(set.len(), count);
            for idx in indices {
                for (i, p) in idx.iter().enumerate() {
                    assert!(p < tensor_data.shape[i]);
                }
            }
        }
    }

    #[test]
    fn idx_in_set_test() -> () {
        let idx1 = Idx::new(vec![1, 2]);
        let idx2 = Idx::new(vec![1, 2]);
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![3, 5]);
        let strides = Strides::new(vec![5, 1]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new(vec![3, 5]), tensor.shape);
        assert_eq!(5, tensor.strides.position(&Idx::new(vec![1, 0])));
        assert_eq!(7, tensor.strides.position(&Idx::new(vec![1, 2])));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![5, 3]);
        let strides = Strides::new(vec![1, 5]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new(vec![5, 3]), tensor.shape);
    }

    #[test]
    fn rand_test() {
        let rand = TensorData::rand(Shape::new(vec![2]));
        println!("rand {:?}", rand);
        assert!(rand.data[0] != rand.data[1]);
    }
}
