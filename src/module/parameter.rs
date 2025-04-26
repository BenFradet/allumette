#[derive(Clone, Debug)]
pub struct Parameter<'a, A> {
    pub name: &'a str,
    pub a: A,
}

impl<'a, A> Parameter<'a, A> {
    pub fn new(name: &'a str, a: A) -> Self {
        Self { name, a }
    }
}
