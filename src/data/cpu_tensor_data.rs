use std::{ops::Index, slice::Iter, sync::Arc};

use proptest::{collection, prelude::*};

use crate::shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides};

use super::tensor_data::TensorData;

#[derive(Clone, Debug)]
pub struct CpuTensorData {
    pub data: Arc<Vec<f64>>,
    pub shape: Shape,
    pub strides: Strides,
}

impl CpuTensorData {
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

    pub fn arbitrary() -> impl Strategy<Value = Self> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f64..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: Strides = (&shape).into();
                Self::new(data, shape, strides)
            })
    }
}

impl Index<Idx> for CpuTensorData {
    type Output = f64;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[self.strides.position(&index)]
    }
}

impl TensorData for CpuTensorData {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    fn iter(&self) -> Iter<'_, f64> {
        self.data.iter()
    }

    fn first(&self) -> Option<f64> {
        self.data.first().copied()
    }

    fn is_contiguous(&self) -> bool {
        if self.strides.is_empty() {
            false
        } else {
            let mut last = self.strides[0];
            for stride in self.strides.iter() {
                if stride > last {
                    return false;
                }
                last = stride;
            }
            true
        }
    }

    fn reshape(&self, shape: Shape) -> Self {
        let strides = (&shape).into();
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
        }
    }

    fn permute(&self, order: &Self) -> Option<Self> {
        let n = self.shape.data().len();
        let ord = Order::from(order);
        if ord.fits(n) {
            let mut new_shape = vec![0; n];
            let mut new_strides = vec![0; n];
            for (idx, value) in ord.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                new_strides[idx] = self.strides[value];
            }
            Some(Self {
                data: Arc::clone(&self.data),
                shape: Shape::new(new_shape),
                strides: Strides::new(new_strides),
            })
        } else {
            None
        }
    }

    fn ones(shape: Shape) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn zeros(shape: Shape) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn rand(shape: Shape) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..shape.size).map(|_| rng.gen()).collect();
        let strides = (&shape).into();
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    fn scalar(s: f64) -> Self {
        let shape = Shape::new(vec![1]);
        let strides = (&shape).into();
        Self {
            data: Arc::new(vec![s]),
            shape,
            strides,
        }
    }

    fn vec(v: Vec<f64>) -> Self {
        let shape = Shape::new(vec![v.len()]);
        let strides = (&shape).into();
        Self {
            data: Arc::new(v),
            shape,
            strides,
        }
    }

    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized,
    {
        if m.is_empty() {
            None
        } else {
            let rows = m[0].len();
            if !m.iter().all(|v| v.len() == rows) {
                None
            } else {
                let cols = m.len();
                let shape = Shape::new(vec![cols, rows]);
                let strides = (&shape).into();
                Some(Self {
                    data: Arc::new(m.concat()),
                    shape,
                    strides,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn assert_tensor_eq(t1: &CpuTensorData, t2: &CpuTensorData) -> () {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.data, t2.data);
    }

    proptest! {
        #[test]
        fn zeros_test(shape in Shape::arbitrary()) {
            let zeros = CpuTensorData::zeros(shape.clone());
            assert_eq!(shape.size, zeros.data.len());
            assert!(zeros.data.iter().all(|f| *f == 0.));
        }

        #[test]
        fn enumeration_test(tensor_data in CpuTensorData::arbitrary()) {
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

        #[test]
        fn permute_test(tensor_data in CpuTensorData::arbitrary(), idx in Idx::arbitrary()) {
            let reversed_index = idx.clone().reverse();
            let pos = tensor_data.strides.position(&idx);
            let order = Order::range(tensor_data.shape.data().len()).reverse();
            let order_td = TensorData::vec(order.data.iter().map(|u| *u as f64).collect());
            let perm_opt = tensor_data.permute(&order_td);
            assert!(perm_opt.is_some());
            let perm = perm_opt.unwrap();
            assert_eq!(pos, perm.strides.position(&reversed_index));
            let orig_opt = perm.permute(&order_td);
            assert!(orig_opt.is_some());
            let orig = orig_opt.unwrap();
            assert_eq!(pos, orig.strides.position(&idx));
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
        let tensor = CpuTensorData::new(data, shape, strides);
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
        let tensor = CpuTensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new(vec![5, 3]), tensor.shape);
    }

    #[test]
    fn rand_test() {
        let rand = CpuTensorData::rand(Shape::new(vec![2]));
        println!("rand {:?}", rand);
        assert!(rand.data[0] != rand.data[1]);
    }
}
