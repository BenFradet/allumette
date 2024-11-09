use crate::tensor::{add, mul, neg};

fn map<I, F, A, B>(i: I, f: F) -> impl Iterator<Item = B>
where 
    I: IntoIterator<Item = A>,
    F: Fn(A) -> B,
{
    let mut result = Vec::new();
    for item in i.into_iter() {
        result.push(f(item));
    }
    result.into_iter()
}

fn zipWith<I1, I2, F, A, B, C>(i1: I1, i2: I2, f: F) -> impl Iterator<Item = C>
where 
    I1: IntoIterator<Item = A>,
    I2: IntoIterator<Item = B>,
    F: Fn(A, B) -> C,
{
    let mut result = Vec::new();
    let mut iter1 = i1.into_iter();
    let mut iter2 = i2.into_iter();
    loop {
        match (iter1.next(), iter2.next()) {
            (Some(a), Some(b)) => result.push(f(a, b)),
            _ => break,
        }
    }
    result.into_iter()
}

fn reduce<I, F, A, B>(i: I, f: F, z: B) -> B
where
    I: IntoIterator<Item = A>,
    F: Fn(A, B) -> B,
{
    let mut result = z;
    for item in i.into_iter() {
        result = f(item, result);
    }
    result
}

fn negs<I>(i: I) -> impl Iterator<Item = f64> where I: IntoIterator<Item = f64> {
    map(i, neg)
}

fn adds<I>(i1: I, i2: I) -> impl Iterator<Item = f64> where I: IntoIterator<Item = f64> {
    zipWith(i1, i2, add)
}

fn sums<I>(i: I) -> f64 where I: IntoIterator<Item = f64> {
    reduce(i, add, 0.)
}

fn prods<I>(i: I) -> f64 where I: IntoIterator<Item = f64> {
    reduce(i, mul, 1.)
}