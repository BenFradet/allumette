use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline};

use crate::{
    backend::{backend::TensorBackend, backend_type::Gpu},
    data::gpu_tensor_data::GpuTensorData,
};

impl TensorBackend<f32, Gpu> for GpuTensorData<'_> {
    fn map<F: Fn(f32) -> f32 + Sync>(&self, _f: F, _tag: &str) -> Self {
        todo!()
    }

    fn map_broadcast<F: Fn(f32) -> f32 + Sync>(&self, out: &Self, f: F, tag: &str) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn zip<F: Fn(f32, f32) -> f32 + Sync>(&self, _other: &Self, _f: F, _tag: &str) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn reduce<F: Fn(f32, f32) -> f32 + Sync>(&self, _f: F, _dim: usize, _init: f64) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }

    fn matmul(&self, _other: &Self) -> Self {
        todo!()
    }
}

fn create_bind_group(
    a: GpuTensorData<'_>,
    b: GpuTensorData<'_>,
    operation: &str,
    pipeline: &ComputePipeline,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    a.device().create_bind_group(&BindGroupDescriptor {
        label: Some("map bind group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: b
                    .create_output_buffer(operation, BufferUsages::STORAGE | BufferUsages::COPY_DST)
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: a.create_shape_buffer().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: b.create_shape_buffer().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: a.create_strides_buffer().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: b.create_strides_buffer().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: a.create_index_buffer().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: b.create_index_buffer().as_entire_binding(),
            },
        ],
    })
}

fn encode(a: GpuTensorData<'_>, pipeline: &ComputePipeline, bind_group: &BindGroup) {
    let mut encoder = a.device().create_command_encoder(&CommandEncoderDescriptor { label: None });
    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, Some(bind_group), &[]);
    let workgroup_number = 12; // TODO: make configurable
    compute_pass.dispatch_workgroups(workgroup_number, 1, 1);
}