use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng, thread_rng};

use crate::backend::backend::Backend;
use crate::shaping::shape::Shape;
use crate::storage::data::Data;
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

    pub fn simple(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
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

    pub fn diag(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
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

    pub fn split(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
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

    pub fn xor(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
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

    pub fn circle(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
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

    pub fn star(n: usize, seed: Option<u64>) -> Self {
        let x = Self::make_points(n, seed);
        let mut y = vec![];
        // all .x5 are wrong
        let star = [
            (E::fromf(0.5), E::fromf(1.)),
            (E::fromf(0.6), E::fromf(0.67)),
            (E::fromf(1.), E::fromf(0.67)),
            (E::fromf(0.67), E::fromf(0.45)),
            (E::fromf(0.85), E::fromf(0.)),
            (E::fromf(0.5), E::fromf(0.25)),
            (E::fromf(0.15), E::fromf(0.)),
            (E::fromf(0.33), E::fromf(0.45)),
            (E::fromf(0.), E::fromf(0.67)),
            (E::fromf(0.4), E::fromf(0.67)),
        ];
        for (x1, x2) in &x {
            let y1 = if Self::in_polygon(*x1, *x2, &star) {
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

    fn in_polygon(x: E, y: E, polygon: &[(E, E)]) -> bool {
        let num_vertices = polygon.len();
        let mut inside = false;
        let mut p1 = polygon[0];

        for i in 1..=num_vertices {
            let p2 = polygon[i % num_vertices];

            // the point is above the min y
            if y > p1.1.min(p2.1) &&
                // the point is below the max y
                y <= p1.1.max(p2.1) &&
                // the point is to the left of the max x
                x <= p1.0.max(p2.0)
            {
                // intersection of the line connecting (x, y) to the edge
                let x_intersection = (y - p1.1) * (p2.0 - p1.0) / (p2.1 - p1.1) + p1.0;

                // the point is on the same line as the edge or to the left of x_intersection
                if p1.0 == p2.0 || x <= x_intersection {
                    inside = !inside;
                }
            }
            p1 = p2;
        }

        inside
    }

    fn make_points(n: usize, seed: Option<u64>) -> Vec<(E, E)> {
        let mut res = Vec::with_capacity(n);
        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(thread_rng()),
        };
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

    #[test]
    fn seed_is_deterministic() {
        let a: Dataset<f64> = Dataset::circle(100, Some(42));
        let b: Dataset<f64> = Dataset::circle(100, Some(42));
        assert_eq!(a.features, b.features);
        assert_eq!(a.labels, b.labels);
    }

    proptest! {
        #[test]
        fn simple_test(n in 0usize..10) {
            let res: Dataset<f64> = Dataset::simple(n, None);
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
            let res: Dataset<f64> = Dataset::diag(n, None);
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
            let res: Dataset<f64> = Dataset::split(n, None);
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
            let res: Dataset<f64> = Dataset::xor(n, None);
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
            let res: Dataset<f64> = Dataset::circle(n, None);
            common_test(&res, n);
            let center = 0.5;
            let radius_sq = 0.20;
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
            let res: Vec<(f64, f64)> = Dataset::make_points(n, None);
            assert_eq!(n, res.len());
            assert!(res.iter().all(|(x1, x2)| *x1 >= 0. && *x1 <= 1. && *x2 >= 0. && *x2 <= 1.));
        }
    }
}
