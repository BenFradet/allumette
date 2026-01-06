pub struct TensorId {
    state: u32,
}

impl TensorId {
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn random(&mut self) -> u32 {
        self.state += 1;
        self.state
    }
}
