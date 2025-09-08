use proptest::{collection, prelude::*};
use rand::Rng;
use std::sync::Arc;

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    wgt::PollType,
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode,
};

use crate::{
    data::tensor_data::TensorData,
    shaping::{iter::Iter, order::Order, shape::Shape, strides::Strides},
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
            label: Some("Tensor new"),
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
        let staging_buffer =
            self.create_output_buffer("to_cpu", BufferUsages::MAP_READ | BufferUsages::COPY_DST);
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

    pub fn arbitrary() -> impl Strategy<Value = Self> {
        Shape::arbitrary().prop_flat_map(Self::arbitrary_with_shape)
    }

    pub fn arbitrary_with_shape(shape: Shape) -> impl Strategy<Value = Self> {
        let size = shape.size;
        let strides: Strides = (&shape).into();
        collection::vec(0.0f32..1., size).prop_map(move |data| {
            Self::new(&data, shape.clone(), strides.clone(), get_wgpu_context())
        })
    }

    fn create_output_buffer(&self, operation: &str, usage: BufferUsages) -> Buffer {
        let size = Self::byte_size(self.shape.size);
        self.context.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Tensor {operation}")),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    fn create_shape_buffer(&self) -> Buffer {
        self.create_storage_buffer(self.shape.iter(), "shape")
    }

    pub fn create_strides_buffer(&self) -> Buffer {
        self.create_storage_buffer(self.strides.iter(), "strides")
    }

    fn create_storage_buffer(&self, iter: Iter<'_>, label: &str) -> Buffer {
        let data: Vec<_> = iter.map(|u| u32::try_from(u).unwrap()).collect();
        self.context
            .device
            .create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: BufferUsages::STORAGE,
            })
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

    fn to_order(&self) -> Order {
        self.into()
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
    use std::collections::HashSet;

    use crate::shaping::idx::Idx;

    use super::*;

    fn assert_tensor_eq(t1: &GpuTensorData, t2: &GpuTensorData) {
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.strides, t2.strides);
        assert_eq!(t1.to_cpu(), t2.to_cpu());
    }

    proptest! {
        #[test]
        fn zeros_test(shape in Shape::arbitrary()) {
            let zeros = GpuTensorData::zeros(shape.clone());
            let zeros_cpu = zeros.to_cpu();
            assert_eq!(shape.size, zeros_cpu.len());
            assert!(zeros_cpu.iter().all(|f| *f == 0.));
        }

        #[test]
        fn enumeration_test(tensor_data in GpuTensorData::arbitrary()) {
            let indices: Vec<_> = tensor_data.indices().collect();
            let count = indices.len();
            assert_eq!(tensor_data.size(), count);
            let set: HashSet<_> = indices.clone().into_iter().collect();
            assert_eq!(set.len(), count);
            for idx in indices {
                for (i, p) in idx.iter().enumerate() {
                    assert!(p < tensor_data.shape[i]);
                }
            }
        }

        #[test]
        fn permute_test(tensor_data in GpuTensorData::arbitrary(), idx in Idx::arbitrary()) {
            let reversed_index = idx.clone().reverse();
            let pos = tensor_data.strides.position(&idx);
            let order = Order::range(tensor_data.shape.data().len()).reverse();
            let order_td = TensorData::from_1d(&order.data.iter().map(|u| *u as f32).collect::<Vec<_>>());
            let perm_opt = tensor_data.permute(&order_td);
            assert!(perm_opt.is_some());
            let perm = perm_opt.unwrap();
            assert_eq!(pos, perm.strides.position(&reversed_index));
            let orig_opt = perm.permute(&order_td);
            assert!(orig_opt.is_some());
            let orig = orig_opt.unwrap();
            assert_eq!(pos, orig.strides.position(&idx));
        }
    }

    #[test]
    fn layout_test1() {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![3, 5]);
        let strides = Strides::new(vec![5, 1]);
        let tensor = GpuTensorData::new(&data, shape, strides, get_wgpu_context());
        assert!(tensor.is_contiguous());
        assert_eq!(Shape::new(vec![3, 5]), tensor.shape);
        assert_eq!(5, tensor.strides.position(&Idx::new(vec![1, 0])));
        assert_eq!(7, tensor.strides.position(&Idx::new(vec![1, 2])));
    }

    #[test]
    fn layout_test2() {
        let data = vec![0.; 15];
        let shape = Shape::new(vec![5, 3]);
        let strides = Strides::new(vec![1, 5]);
        let tensor = GpuTensorData::new(&data, shape, strides, get_wgpu_context());
        assert!(!tensor.is_contiguous());
        assert_eq!(Shape::new(vec![5, 3]), tensor.shape);
    }

    #[test]
    fn rand_test() {
        let rand = GpuTensorData::rand(Shape::new(vec![2]));
        let rand_cpu = rand.to_cpu();
        assert!(rand_cpu[0] != rand_cpu[1]);
    }
}
