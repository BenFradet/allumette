pub trait TensorBackendType {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;
#[derive(Debug, Clone)]
pub struct Gpu;

impl TensorBackendType for Seq {}
impl TensorBackendType for Par {}
impl TensorBackendType for Gpu {}

pub trait BackendType = TensorBackendType + Clone + std::fmt::Debug;
