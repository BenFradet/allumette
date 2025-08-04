use std::{slice::Iter, sync::Arc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    wgt::PollType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, MapMode, Queue,
};

use crate::{
    data::tensor_data::TensorData,
    shaping::{idx::Idx, order::Order, shape::Shape, strides::Strides},
};

#[derive(Clone, Debug)]
pub struct GpuTensorData<'a> {
    buffer: Arc<wgpu::Buffer>,
    pub shape: Shape,
    pub strides: Strides,
    device: &'a Device,
    queue: &'a Queue,
}

const WGPU_ELEMENT_SIZE: usize = std::mem::size_of::<f32>();

impl<'a> GpuTensorData<'a> {
    pub fn new(
        data: &[f32],
        shape: Shape,
        strides: Strides,
        device: &'a Device,
        queue: &'a Queue,
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("new gpu tensor data"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        Self {
            buffer: Arc::new(buffer),
            shape,
            strides,
            device,
            queue,
        }
    }

    // see repeated_compute example in wgpu
    pub fn to_cpu(&self) -> Vec<f32> {
        let size = Self::byte_size(self.shape.size);
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(PollType::wait()).unwrap();

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
        todo!()
    }

    fn index(&self, idx: Idx) -> f32 {
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
        Self: Sized,
    {
        todo!()
    }

    fn transpose(&self) -> Option<Self> {
        let mut order: Vec<_> = Order::range(self.shape().len())
            .data
            .iter()
            .map(|&u| u as f32)
            .collect();
        let len = order.len();
        order.swap(len - 2, len - 1);
        self.permute(&Self::vec(order))
    }

    fn indices(&self) -> impl Iterator<Item = crate::shaping::idx::Idx> {
        std::iter::empty()
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

    fn epsilon(shape: Shape, idx: &crate::shaping::idx::Idx, eps: f32) -> Self {
        todo!()
    }

    fn from(data: Vec<f32>, shape: Shape, strides: Strides) -> Self {
        todo!()
    }

    fn scalar(s: f32) -> Self {
        todo!()
    }

    fn vec(v: Vec<f32>) -> Self {
        todo!()
    }

    fn matrix(m: Vec<Vec<f32>>) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
