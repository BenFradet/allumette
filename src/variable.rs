use std::collections::HashMap;

pub trait Variable: Sized {
    fn accumulate_derivative(self, x: f64) -> Self;
    fn id(&self) -> u64;
    fn is_leaf(&self) -> bool;
    fn is_constant(&self) -> bool;
    fn parents(&self) -> impl Iterator<Item = &Self>;
    fn chain_rule(&self, d: f64) -> impl Iterator<Item = (&Self, f64)>;
    fn topological_sort(&self) -> impl Iterator<Item = &Self>;
    fn backprop(&self, d: f64) -> HashMap<u64, Self>;
}
