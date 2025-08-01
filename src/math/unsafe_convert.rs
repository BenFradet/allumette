pub trait UnsafeToUsize {
    fn unsafe_convert(&self) -> usize;
}

impl UnsafeToUsize for f32 {
    fn unsafe_convert(&self) -> usize {
        *self as usize
    }
}

impl UnsafeToUsize for f64 {
    fn unsafe_convert(&self) -> usize {
        *self as usize
    }
}