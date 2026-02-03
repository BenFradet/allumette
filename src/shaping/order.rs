use std::collections::HashSet;

use crate::storage::{cpu_data::CpuData, gpu_data::GpuData};

use super::iter::Iter;

#[derive(Clone, Debug)]
pub struct Order {
    pub data: Vec<usize>,
}

impl Order {
    pub fn new(data: Vec<usize>) -> Option<Self> {
        let len = data.len();
        let s = Self { data };
        if s.fits(len) { Some(s) } else { None }
    }

    pub fn range(n: usize) -> Self {
        Self {
            data: (0..n).collect::<Vec<_>>(),
        }
    }

    pub fn reverse(mut self) -> Self {
        self.data.reverse();
        self
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter::new(&self.data)
    }

    // TODO: refactor
    pub fn fits(&self, n: usize) -> bool {
        let s1: HashSet<_> = self.data.iter().copied().collect();
        let s2: HashSet<_> = (0..n).collect();
        s1 == s2
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn index(&self, n: usize) -> Option<usize> {
        self.data.iter().position(|i| *i == n)
    }
}

impl From<CpuData> for Order {
    fn from(td: CpuData) -> Self {
        let ord_data = td.data.clone();
        let len = ord_data.len();
        Order::new(ord_data.iter().map(|f| *f as usize).collect()).unwrap_or(Order::range(len))
    }
}

impl From<&CpuData> for Order {
    fn from(td: &CpuData) -> Self {
        let ord_data = td.data.clone();
        let len = ord_data.len();
        Order::new(ord_data.iter().map(|f| *f as usize).collect()).unwrap_or(Order::range(len))
    }
}

impl From<GpuData<'_>> for Order {
    fn from(td: GpuData) -> Self {
        let ord_data = td.to_cpu();
        let len = ord_data.len();
        Order::new(ord_data.iter().map(|f| *f as usize).collect()).unwrap_or(Order::range(len))
    }
}

impl From<&GpuData<'_>> for Order {
    fn from(td: &GpuData) -> Self {
        let ord_data = td.to_cpu();
        let len = ord_data.len();
        Order::new(ord_data.iter().map(|f| *f as usize).collect()).unwrap_or(Order::range(len))
    }
}
