pub struct ConstIter<'a, const N: usize> {
    data: &'a [usize; N],
    index: usize,
}

impl<'a, const N: usize> ConstIter<'a, N> {
    pub fn new(data: &'a [usize; N]) -> Self {
        Self {
            data,
            index: 0,
        }
    }
}

impl<const N: usize> Iterator for ConstIter<'_, N> {
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