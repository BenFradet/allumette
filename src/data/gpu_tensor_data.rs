use rand::Rng;
use std::sync::Arc;

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    wgt::PollType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode,
};

use crate::{
    data::tensor_data::TensorData,
    shaping::{order::Order, shape::Shape, strides::Strides},
    wgpu::wgpu_context::{get_wgpu_context, WgpuContext},
};

#[derive(Clone, Debug)]
pub struct GpuTensorData<'a> {
    buffer: Arc<wgpu::Buffer>,
    pub shape: Shape,
    pub strides: Strides,
    context: &'a WgpuContext,
}

const WGPU_ELEMENT_SIZE: usize = std::mem::size_of::<f32>();

impl<'a> GpuTensorData<'a> {
    pub fn new(data: &[f32], shape: Shape, strides: Strides, context: &'a WgpuContext) -> Self {
        let buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("new gpu tensor data"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        Self {
            buffer: Arc::new(buffer),
            shape,
            strides,
            context,
        }
    }

    // see repeated_compute example in wgpu
    pub fn to_cpu(&self) -> Vec<f32> {
        let size = Self::byte_size(self.shape.size);
        let staging_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size);

        self.context.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |r| sender.send(r).unwrap());
        self.context.device.poll(PollType::wait()).unwrap();

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        } else {
            panic!("failed to read buffer from GPU: BufferAsyncError");
        }
    }

    fn byte_size(size: usize) -> u64 {
        u64::try_from(size * WGPU_ELEMENT_SIZE).unwrap()
    }
}

impl TensorData<f32> for GpuTensorData<'_> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn size(&self) -> usize {
        self.shape.size
    }

    fn collect(&self) -> Vec<f32> {
        self.to_cpu()
    }

    fn first(&self) -> Option<f32> {
        self.to_cpu().first().copied()
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
        let strides = (&shape).into();
        Self {
            buffer: Arc::clone(&self.buffer),
            shape,
            strides,
            context: self.context,
        }
    }

    fn permute(&self, order: &Self) -> Option<Self>
    where
        Self: Sized,
    {
        let n = self.shape.data().len();
        let ord = Order::from(order);
        if ord.fits(n) {
            let mut new_shape = vec![0; n];
            let mut new_strides = vec![0; n];
            for (idx, value) in ord.iter().enumerate() {
                new_shape[idx] = self.shape[value];
                new_strides[idx] = self.strides[value];
            }
            Some(Self {
                buffer: Arc::clone(&self.buffer),
                shape: Shape::new(new_shape),
                strides: Strides::new(new_strides),
                context: self.context,
            })
        } else {
            None
        }
    }

    fn transpose(&self) -> Option<Self> {
        let mut order: Vec<_> = Order::range(self.shape().len())
            .data
            .iter()
            .map(|&u| u as f32)
            .collect();
        let len = order.len();
        order.swap(len - 2, len - 1);
        self.permute(&Self::from_1d(&order))
    }

    fn indices(&self) -> impl Iterator<Item = crate::shaping::idx::Idx> {
        (0..self.size()).map(|i| self.strides.idx(i))
    }

    fn ones(shape: Shape) -> Self {
        let data = vec![1.; shape.size];
        let strides = (&shape).into();
        Self::new(&data, shape, strides, get_wgpu_context())
    }

    fn zeros(shape: Shape) -> Self {
        let data = vec![0.; shape.size];
        let strides = (&shape).into();
        Self::new(&data, shape, strides, get_wgpu_context())
    }

    fn rand(shape: Shape) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.size).map(|_| rng.gen()).collect();
        let strides = (&shape).into();
        Self::new(&data, shape, strides, get_wgpu_context())
    }

    fn epsilon(shape: Shape, idx: &crate::shaping::idx::Idx, eps: f32) -> Self {
        let strides: Strides = (&shape).into();
        let mut data = vec![0.; shape.size];
        data[strides.position(idx)] = eps;
        Self::new(&data, shape, strides, get_wgpu_context())
    }

    fn from(data: &[f32], shape: Shape, strides: Strides) -> Self {
        Self::new(data, shape, strides, get_wgpu_context())
    }

    fn from_scalar(s: f32) -> Self {
        let shape = Shape::new(vec![1]);
        let strides = (&shape).into();
        Self::new(&[s], shape, strides, get_wgpu_context())
    }

    fn from_1d(v: &[f32]) -> Self {
        let shape = Shape::new(vec![v.len()]);
        let strides = (&shape).into();
        Self::new(v, shape, strides, get_wgpu_context())
    }

    fn from_2d(m: &[&[f32]]) -> Option<Self>
    where
        Self: Sized,
    {
        if m.is_empty() {
            None
        } else {
            let rows = m[0].len();
            if !m.iter().all(|v| v.len() == rows) {
                None
            } else {
                let cols = m.len();
                let shape = Shape::new(vec![cols, rows]);
                let strides = (&shape).into();
                Some(Self::new(&m.concat(), shape, strides, get_wgpu_context()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_eq(t1: &GpuTensorData, t2: &GpuTensorData) {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.to_cpu(), t2.to_cpu());
    }
}