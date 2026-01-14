pub trait TensorMode {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;
#[derive(Debug, Clone)]
pub struct Gpu;

impl TensorMode for Seq {}
impl TensorMode for Par {}
impl TensorMode for Gpu {}

pub trait Mode = TensorMode + Clone + std::fmt::Debug;
