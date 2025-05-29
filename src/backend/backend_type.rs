pub trait TensorBackendType {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;

impl TensorBackendType for Seq {}
impl TensorBackendType for Par {}

pub trait BackendType = TensorBackendType + Clone + std::fmt::Debug;
