pub trait Variable {
    fn derivative(&self) -> Option<f64>;
    fn id(&self) -> u64;
    fn is_leaf(&self) -> bool;
    fn is_constant(&self) -> bool;
}