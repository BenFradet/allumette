pub trait BackendType {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;

impl BackendType for Seq {}
impl BackendType for Par {}
