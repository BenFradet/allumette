use proptest::{collection, prelude::*};

use super::{idx::Idx, order::Order, shape::Shape, strides::Strides};

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

    fn position(&self, idx: &Idx<N>) -> usize {
        idx.iter()
            .zip(self.strides.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride)
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    #[allow(clippy::needless_range_loop)]
    fn index(&self, pos: usize) -> Idx<N> {
        let mut res = [1; N];
        let mut mut_pos = pos;
        for i in 0..N {
            let s = self.strides[i];
            let idx = mut_pos / s;
            mut_pos -= idx * s;
            res[i] = idx;
        }
        Idx::new(res)
    }

    // TODO: look into use<'_, N>
    fn indices(&self) -> impl Iterator<Item = Idx<N>> + use<'_, N> {
        (0..self.size()).map(|i| self.index(i))
    }

    fn permute(mut self, order: &Order<N>) -> Option<Self> {
        if order.fits() {
            let mut new_shape = [0; N];
            let mut new_strides = [0; N];
            for (idx, value) in order.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                //new_strides[idx] = self.strides[value];
            }
            self.shape = Shape::new(new_shape);
            //self.strides = Strides::new(new_strides);
            self.strides = (&self.shape).into();
            Some(self)
        } else {
            None
        }
    }

    fn is_contiguous(&self) -> bool {
        let res = self
            .strides
            .iter()
            .fold((true, usize::MAX), |(is_contiguous, last), stride| {
                if !is_contiguous || stride > last {
                    (false, stride)
                } else {
                    (true, stride)
                }
            });
        res.0
    }

    fn arbitrary() -> impl Strategy<Value = TensorData<N>> {
        Shape::arbitrary()
            .prop_flat_map(|shape| {
                let size = shape.size;
                let data = collection::vec(0.0f64..1., size);
                (data, Just(shape))
            })
            .prop_map(|(data, shape)| {
                let strides: Strides<N> = (&shape).into();
                TensorData::new(data, shape, strides)
            })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn permute_manual() -> () {
        let shape = Shape::new([2, 1, 2, 2]);
        let strides = (&shape).into();
        println!("strides: {:?}", strides);
        //let ind = Idx::new([0, 0, 1, 1]);
        let td = TensorData::new(vec![], shape, strides);
        let order = Order::range().reverse();
        let perm = td.permute(&order);
        println!("perm: {:?}", perm);
    }

    proptest! {
        // TODO: find a way to have arbitrary const generics?

        #[test]
        fn permute_test(tensor_data in TensorData::<4>::arbitrary(), idx in Idx::<4>::arbitrary()) {
            let pos = tensor_data.position(&idx);
            let order = Order::range().reverse();
            println!("idx: {:?}", idx);
            println!("orig: {:?}", tensor_data);
            let perm_opt = tensor_data.permute(&order);
            assert!(perm_opt.is_some());
            let perm = perm_opt.unwrap();
            println!("perm: {:?}", perm);
            assert_eq!(pos, perm.position(&idx));
            let orig_opt = perm.permute(&order);
            assert!(orig_opt.is_some());
            let orig = orig_opt.unwrap();
            println!("orig: {:?}", orig);
            println!();

            assert_eq!(pos, orig.position(&idx));
        }

        #[test]
        fn position_test(tensor_data in TensorData::<4>::arbitrary()) {
            for idx in tensor_data.indices() {
                let pos = tensor_data.position(&idx);
                assert!(pos < tensor_data.size());
            }
        }

        #[test]
        fn enumeration_test(tensor_data in TensorData::<4>::arbitrary()) {
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
        let idx1 = Idx::new([1, 2]);
        let idx2 = Idx::new([1, 2]);
        let mut set = HashSet::new();
        set.insert(idx1);
        let res = set.insert(idx2);
        assert!(!res);
        assert_eq!(1, set.len());
    }

    #[test]
    fn layout_test1() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([3, 5]);
        let strides = Strides::new([5, 1]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new([3, 5]), tensor.shape);
        assert_eq!(5, tensor.position(&Idx::new([1, 0])));
        assert_eq!(7, tensor.position(&Idx::new([1, 2])));
    }

    #[test]
    fn layout_test2() -> () {
        let data = vec![0.; 15];
        let shape = Shape::new([5, 3]);
        let strides = Strides::new([1, 5]);
        let tensor = TensorData::new(data, shape, strides);
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new([5, 3]), tensor.shape);
    }
}
