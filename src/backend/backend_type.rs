pub trait BT {}

#[derive(Debug, Clone)]
pub struct Seq;
#[derive(Debug, Clone)]
pub struct Par;

impl BT for Seq {}
impl BT for Par {}

pub trait BackendType = BT + Clone + std::fmt::Debug;
