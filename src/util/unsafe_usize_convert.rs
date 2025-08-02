pub trait UnsafeUsizeConvert {
    fn unsafe_to(&self) -> usize;
    fn unsafe_from(u: usize) -> Self;
}

impl UnsafeUsizeConvert for f32 {
    fn unsafe_to(&self) -> usize {
        *self as usize
    }

    fn unsafe_from(u: usize) -> Self {
        u as f32
    }
}

impl UnsafeUsizeConvert for f64 {
    fn unsafe_to(&self) -> usize {
        *self as usize
    }

    fn unsafe_from(u: usize) -> Self {
        u as f64
    }
}