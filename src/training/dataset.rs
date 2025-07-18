use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct Dataset {
    pub n: usize,
    pub x: Vec<(f32, f32)>,
    pub y: Vec<usize>,
}

impl Dataset {
    pub fn simple(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for v in &x {
            let y1 = if v.0 < 0.5 { 1 } else { 0 };
            y.push(y1);
        }
        Self { n, x, y }
    }

    pub fn diag(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for (x1, x2) in &x {
            let y1 = if x1 + x2 < 0.5 { 1 } else { 0 };
            y.push(y1);
        }
        Self { n, x, y }
    }

    pub fn split(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for v in &x {
            let y1 = if v.0 < 0.2 || v.0 > 0.8 { 1 } else { 0 };
            y.push(y1);
        }
        Self { n, x, y }
    }

    pub fn xor(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        for (x1, x2) in &x {
            let y1 = if (*x1 < 0.5 && *x2 > 0.5) || (*x1 > 0.5 && *x2 < 0.5) {
                1
            } else {
                0
            };
            y.push(y1);
        }
        Self { n, x, y }
    }

    pub fn circle(n: usize) -> Self {
        let x = Self::make_points(n);
        let mut y = vec![];
        let center = 0.5;
        let radius_sq = 0.25;
        for (x1, x2) in &x {
            let (x1p, x2p) = (x1 - center, x2 - center);
            let y1 = if x1p * x1p + x2p * x2p < radius_sq {
                1
            } else {
                0
            };
            y.push(y1);
        }
        Self { n, x, y }
    }

    fn make_points(n: usize) -> Vec<(f32, f32)> {
        let mut res = vec![];
        let mut rng = thread_rng();
        for _i in 0..n {
            let x1 = rng.gen();
            let x2 = rng.gen();
            res.push((x1, x2));
        }
        res
    }
}

use proptest::prelude::*;

fn common_test(ds: &Dataset, n: usize) {
    assert_eq!(n, ds.n);
    assert_eq!(n, ds.x.len());
    assert_eq!(n, ds.y.len());
    assert!(ds
        .x
        .iter()
        .all(|(x1, x2)| *x1 >= 0. && *x1 <= 1. && *x2 >= 0. && *x2 <= 1.));
    assert!(ds.y.iter().all(|y| *y == 0 || *y == 1));
}

proptest! {
    #[test]
    fn simple_test(n in 0usize..10) {
        let res = Dataset::simple(n);
        common_test(&res, n);
        assert!(res.x.iter().zip(res.y.iter()).all(|((x1, _x2), y)| {
            if *x1 < 0.5 {
                *y == 1
            } else {
                *y == 0
            }
        }));
    }

    #[test]
    fn diag_test(n in 0usize..10) {
        let res = Dataset::diag(n);
        common_test(&res, n);
        assert!(res.x.iter().zip(res.y.iter()).all(|((x1, x2), y)| {
            if *x1 + *x2 < 0.5 {
                *y == 1
            } else {
                *y == 0
            }
        }));
    }

    #[test]
    fn split_test(n in 0usize..10) {
        let res = Dataset::split(n);
        common_test(&res, n);
        assert!(res.x.iter().zip(res.y.iter()).all(|((x1, _x2), y)| {
            if *x1 < 0.2 || *x1 > 0.8 {
                *y == 1
            } else {
                *y == 0
            }
        }));
    }

    #[test]
    fn xor_test(n in 0usize..10) {
        let res = Dataset::xor(n);
        common_test(&res, n);
        assert!(res.x.iter().zip(res.y.iter()).all(|((x1, x2), y)| {
            if (*x1 < 0.5 && *x2 > 0.5) || (*x1 > 0.5 && *x2 < 0.5) {
                *y == 1
            } else {
                *y == 0
            }
        }));
    }

    #[test]
    fn circle_test(n in 0usize..10) {
        let res = Dataset::circle(n);
        common_test(&res, n);
        let center = 0.5;
        let radius_sq = 0.25;
        assert!(res.x.iter().zip(res.y.iter()).all(|((x1, x2), y)| {
            if (x1 - center).powf(2.) + (x2 - center).powf(2.) < radius_sq {
                *y == 1
            } else {
                *y == 0
            }
        }));
    }

    #[test]
    fn make_points_test(n in 0usize..10) {
        let res = Dataset::make_points(n);
        assert_eq!(n, res.len());
        assert!(res.iter().all(|(x1, x2)| *x1 >= 0. && *x1 <= 1. && *x2 >= 0. && *x2 <= 1.));
    }
}
