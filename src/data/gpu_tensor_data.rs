use std::sync::Arc;

use wgpu::{Device, util::BufferInitDescriptor, BufferUsages};

use crate::{data::tensor_data::TensorData, shaping::{shape::Shape, strides::Strides}};

#[derive(Clone, Debug)]
pub struct GpuTensorData<'a> {
    buffer: Arc<wgpu::Buffer>,
    pub shape: Shape,
    pub strides: Strides,
    device: &'a Device,
}

impl<'a> GpuTensorData<'a> {
    pub fn new(data: &[f32], shape: Shape, strides: Strides, device: &'a Device) -> Self {
        let buffer = device.create_buffer_init(BufferInitDescriptor {
            label: Some("new gpu tensor data"),
            contents: data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        Self {
            buffer: Arc::new(buffer),
            shape,
            strides,
            device,
        }
    }
}

impl TensorData for GpuTensorData {
    fn shape(&self) -> &Shape {
        self.shape
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    fn iter(&self) -> std::slice::Iter<'_, f64> {
        todo!()
    }

    fn first(&self) -> Option<f64> {
        todo!()
    }

    // TODO: factor out
    fn is_contiguous(&self) -> bool {
        if self.strides.is_empty() {
            false
        } else {
            let mut last = self.strides[0];
            for stride in self.strides.iter() {
                if stride > last {
                    return false;
                }
                last = stride;
            }
            true
        }
    }

    fn reshape(&self, shape: Shape) -> Self {
        todo!()
    }

    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized {
        todo!()
    }

    fn index(&self, idx: crate::shaping::idx::Idx) -> f64 {
        todo!()
    }

    fn indices(&self) -> impl Iterator<Item = crate::shaping::idx::Idx> {
        todo!()
    }

    fn ones(shape: Shape) -> Self {
        todo!()
    }

    fn zeros(shape: Shape) -> Self {
        todo!()
    }

    fn rand(shape: Shape) -> Self {
        todo!()
    }

    fn epsilon(shape: Shape, idx: &crate::shaping::idx::Idx, eps: f64) -> Self {
        todo!()
    }

    fn from(data: Vec<f64>, shape: Shape, strides: Strides) -> Self {
        todo!()
    }

    fn scalar(s: f64) -> Self {
        todo!()
    }

    fn vec(v: Vec<f64>) -> Self {
        todo!()
    }

    fn matrix(m: Vec<Vec<f64>>) -> Option<Self>
    where
        Self: Sized {
        todo!()
    }
}
