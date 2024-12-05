pub trait Variable: Sized {
    fn derivative(&self) -> Option<f64>;
    fn id(&self) -> u64;
    fn is_leaf(&self) -> bool;
    fn is_constant(&self) -> bool;
    fn parents(&self) -> impl Iterator<Item = &Self>;
    fn chain_rule(&self, d: f64) -> impl Iterator<Item = (&Self, f64)>;
    fn topological_sort(&self) -> impl Iterator<Item = &Self>;
}
