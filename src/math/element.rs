pub trait Element: Clone + std::fmt::Debug + Clone
where
    Self: Sized,
{
    fn zero() -> Self;
    fn one() -> Self;
    fn fromf(f: f64) -> Self;
}

impl Element for f32 {
    fn one() -> Self {
        1.
    }

    fn zero() -> Self {
        0.
    }

    fn fromf(f: f64) -> Self {
        f as f32
    }
}

impl Element for f64 {
    fn one() -> Self {
        1.
    }

    fn zero() -> Self {
        0.
    }

    fn fromf(f: f64) -> Self {
        f
    }
}
