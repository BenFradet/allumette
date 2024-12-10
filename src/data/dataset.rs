use rand::{thread_rng, Rng};

pub struct Dataset {
    pub n: usize,
    pub x: Vec<(f64, f64)>,
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
        for (x1, x2) in &x {
            let (x1p, x2p) = (x1 - 0.5, x2 - 0.5);
            let y1 = if x1p * x1p + x2p * x2p > 0.1 { 1 } else { 0 };
            y.push(y1);
        }
        Self { n, x, y }
    }

    fn make_points(n: usize) -> Vec<(f64, f64)> {
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
