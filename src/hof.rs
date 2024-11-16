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

fn zip_with<I1, I2, F, A, B, C>(i1: I1, i2: I2, f: F) -> impl Iterator<Item = C>
where
    I1: IntoIterator<Item = A>,
    I2: IntoIterator<Item = B>,
    F: Fn(A, B) -> C,
{
    let mut result = Vec::new();
    let mut iter1 = i1.into_iter();
    let mut iter2 = i2.into_iter();
    while let (Some(a), Some(b)) = (iter1.next(), iter2.next()) {
        result.push(f(a, b));
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

fn negs<I>(i: I) -> impl Iterator<Item = f64>
where
    I: IntoIterator<Item = f64>,
{
    map(i, |a| -a)
}

fn adds<I>(i1: I, i2: I) -> impl Iterator<Item = f64>
where
    I: IntoIterator<Item = f64>,
{
    zip_with(i1, i2, |a, b| a + b)
}

fn sums<I>(i: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    reduce(i, |a, b| a + b, 0.)
}

fn prods<I>(i: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    reduce(i, |a, b| a * b, 1.)
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn test_map(array in prop::array::uniform10(0f64..)) {
        let res = map(array, |a| a);
        assert!(array.into_iter().zip(res).all(|(a, b)| a == b));
    }

    #[test]
    fn test_negs(array in prop::array::uniform10(0f64..)) {
        let res = negs(array);
        assert!(res.into_iter().all(|a| a < 0.));
    }

    #[test]
    fn test_adds(array in prop::array::uniform10(0f64..)) {
        let res = adds(array, array);
        assert!(res.zip(array).into_iter().all(|(a, b)| a == 2. * b));
    }

    #[test]
    fn test_sums(array in prop::array::uniform10(0f64..)) {
        let res = sums(array);
        assert_eq!(res, array.into_iter().sum());
    }

    #[test]
    fn test_prods(array in prop::array::uniform10(0f64..)) {
        let res = prods(array);
        assert_eq!(res, array.into_iter().product());
    }
}
