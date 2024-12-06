pub trait Optimizer {
    fn zero(self) -> Self;
    fn step(self) -> Self;
}