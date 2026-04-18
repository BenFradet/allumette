pub trait TensorMode {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct Gpu;

impl TensorMode for Seq {}
impl TensorMode for Par {}
#[cfg(feature = "gpu")]
impl TensorMode for Gpu {}

pub trait Mode = TensorMode + Clone + std::fmt::Debug;
