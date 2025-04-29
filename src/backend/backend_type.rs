pub trait BackendType {}

pub struct Sequential;
pub struct Parallel;

impl BackendType for Sequential {}
impl BackendType for Parallel {}
