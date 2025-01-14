pub struct TypeIf<const COND: bool>;

pub trait TypeTrue {}
impl TypeTrue for TypeIf<true> {}