pub trait Optimizer {
    fn zero(&mut self) -> ();
    fn step(&mut self) -> ();
}
