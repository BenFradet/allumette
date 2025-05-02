pub trait BackendType {}

#[derive(Debug, Clone)]
pub struct Sequential;
#[derive(Debug, Clone)]
pub struct Parallel;

impl BackendType for Sequential {}
impl BackendType for Parallel {}
