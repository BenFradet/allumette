pub trait UnaryOps {
    type Placeholder;

    fn exp(p: &Self::Placeholder) -> Self::Placeholder;
    fn inv(p: &Self::Placeholder) -> Self::Placeholder;
    fn log(p: &Self::Placeholder) -> Self::Placeholder;
    fn neg(p: &Self::Placeholder) -> Self::Placeholder;
    fn relu(p: &Self::Placeholder) -> Self::Placeholder;
    fn sig(p: &Self::Placeholder) -> Self::Placeholder;
}
