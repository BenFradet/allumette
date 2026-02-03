use rand::{Rng, thread_rng};

use crate::backend::backend::Backend;
use crate::storage::data::Data;
use crate::shaping::shape::Shape;
use crate::util::unsafe_usize_convert::UnsafeUsizeConvert;
use crate::{math::element::Element, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Dataset<E: Element> {
    pub n: usize,
    pub features: Vec<(E, E)>,
    pub labels: Vec<usize>,
}

impl<E: Element + UnsafeUsizeConvert> Dataset<E> {
    pub fn features<'a, B: Backend<Element = E>>(&self) -> Tensor<'a, B> {
        Tensor::from_tuples(&self.features)
    }

    pub fn labels<'a, B: Backend<Element = E>>(&self) -> Tensor<'a, B> {
        let label_data = <B::Storage<'a> as Data<E>>::from_1d(
            &self
                .labels
                .iter()
                .map(|u| E::fromf(*u as f64))
                .collect::<Vec<_>>(),
        );
        Tensor::from_data(label_data)
    }

    pub fn n<'a, B: Backend<Element = E>>(&self) -> Tensor<'a, B> {
        Tensor::from_shape(
            &std::iter::repeat_n(E::fromf(self.n as f64), self.n).collect::<Vec<_>>(),
            self.n_shape(),
        )
    }

    pub fn ones<'a, B: Backend>(&self) -> Tensor<'a, B> {
        Tensor::ones(self.n_shape())
    }

    pub fn n_shape(&self) -> Shape {
        Shape::scalar(self.n)
    }

    pub fn simple(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for v in &x {
            let y1 = if v.0 < E::fromf(0.5) { 1 } else { 0 };
            y.push(y1);
        }
        Self {
            n,
            features: x,
            labels: y,
        }
    }

    pub fn diag(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for (x1, x2) in &x {
            let y1 = if *x1 + *x2 < E::fromf(0.5) { 1 } else { 0 };
            y.push(y1);
        }
        Self {
            n,
            features: x,
            labels: y,
        }
    }

    pub fn split(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for v in &x {
            let y1 = if v.0 < E::fromf(0.2) || v.0 > E::fromf(0.8) {
                1
            } else {
                0
            };
            y.push(y1);
        }
        Self {
            n,
            features: x,
            labels: y,
        }
    }

    pub fn xor(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for (x1, x2) in &x {
            let y1 = if (*x1 < E::fromf(0.5) && *x2 > E::fromf(0.5))
                || (*x1 > E::fromf(0.5) && *x2 < E::fromf(0.5))
            {
                1
            } else {
                0
            };
            y.push(y1);
        }
        Self {
            n,
            features: x,
            labels: y,
        }
    }

    pub fn circle(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        let center = E::fromf(0.5);
        let radius_sq = E::fromf(0.20);
        for (x1, x2) in &x {
            let (x1p, x2p) = (*x1 - center, *x2 - center);
            let y1 = if x1p * x1p + x2p * x2p < radius_sq {
                1
            } else {
                0
            };
            y.push(y1);
        }
        Self {
            n,
            features: x,
            labels: y,
        }
    }

    fn make_points(n: usize) -> Vec<(E, E)> {
        let mut res = vec![];
        let mut rng = thread_rng();
        for _i in 0..n {
            let x1 = E::fromf(rng.r#gen());
            let x2 = E::fromf(rng.r#gen());
            res.push((x1, x2));
        }
        res
    }
}

mod tests {
    use super::*;
    use proptest::prelude::*;

    // proptest macro is not picked up
    #[allow(dead_code)]
    fn common_test<E: Element>(ds: &Dataset<E>, n: usize) {
        assert_eq!(n, ds.n);
        assert_eq!(n, ds.features.len());
        assert_eq!(n, ds.labels.len());
        assert!(ds.features.iter().all(|(x1, x2)| *x1 >= E::zero()
            && *x1 <= E::one()
            && *x2 >= E::zero()
            && *x2 <= E::one()));
        assert!(ds.labels.iter().all(|y| *y == 0 || *y == 1));
    }

    proptest! {
        #[test]
        fn simple_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::simple(n);
            common_test(&res, n);
            assert!(res.features.iter().zip(res.labels.iter()).all(|((x1, _x2), y)| {
                if *x1 < 0.5 {
                    *y == 1
                } else {
                    *y == 0
                }
            }));
        }

        #[test]
        fn diag_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::diag(n);
            common_test(&res, n);
            assert!(res.features.iter().zip(res.labels.iter()).all(|((x1, x2), y)| {
                if *x1 + *x2 < 0.5 {
                    *y == 1
                } else {
                    *y == 0
                }
            }));
        }

        #[test]
        fn split_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::split(n);
            common_test(&res, n);
            assert!(res.features.iter().zip(res.labels.iter()).all(|((x1, _x2), y)| {
                if *x1 < 0.2 || *x1 > 0.8 {
                    *y == 1
                } else {
                    *y == 0
                }
            }));
        }

        #[test]
        fn xor_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::xor(n);
            common_test(&res, n);
            assert!(res.features.iter().zip(res.labels.iter()).all(|((x1, x2), y)| {
                if (*x1 < 0.5 && *x2 > 0.5) || (*x1 > 0.5 && *x2 < 0.5) {
                    *y == 1
                } else {
                    *y == 0
                }
            }));
        }

        #[test]
        fn circle_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::circle(n);
            common_test(&res, n);
            let center = 0.5;
            let radius_sq = 0.25;
            assert!(res.features.iter().zip(res.labels.iter()).all(|((x1, x2), y)| {
                if (x1 - center).powf(2.) + (x2 - center).powf(2.) < radius_sq {
                    *y == 1
                } else {
                    *y == 0
                }
            }));
        }

        #[test]
        fn make_points_test(n in 0usize..10) {
            let res: Vec<(f64, f64)> = Dataset::make_points(n);
            assert_eq!(n, res.len());
            assert!(res.iter().all(|(x1, x2)| *x1 >= 0. && *x1 <= 1. && *x2 >= 0. && *x2 <= 1.));
        }
    }
}
