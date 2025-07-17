use std::sync::Arc;

use crate::shaping::{shape::Shape, strides::Strides};

#[derive(Clone, Debug)]
pub struct GpuTensorData {
    buffer: Arc<wgpu::Buffer>,
    pub shape: Shape,
    pub strides: Strides,
}
