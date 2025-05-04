pub struct Iter<'a> {
    data: &'a [usize],
    index: usize,
}

impl<'a> Iter<'a> {
    pub fn new(data: &'a [usize]) -> Self {
        Self { data, index: 0 }
    }
}

impl Iterator for Iter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let res = self.data[self.index];
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }
}
