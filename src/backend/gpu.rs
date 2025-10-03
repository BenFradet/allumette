use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BufferUsages, ComputePipeline};

use crate::{
    backend::{backend::TensorBackend, backend_type::Gpu},
    data::gpu_tensor_data::GpuTensorData,
};

impl TensorBackend<f32, Gpu> for GpuTensorData<'_> {
    // TODO: rm unwrap
    fn map<F: Fn(f32) -> f32 + Sync>(&self, _f: F, tag: &'static str) -> Self {
        let shape = self.shape.clone();
        let gpu_size = shape.gpu_byte_size();
        // make contiguous
        let strides = (&shape).into();

        let buffer = self.context.create_output_buffer(
            gpu_size,
            tag,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let td = GpuTensorData::from_buffer(shape, strides, buffer, self.context);

        let workgroup_info = (&td.shape).into();
        let pipeline = td
            .context
            .get_or_create_pipeline(tag, workgroup_info)
            .unwrap();

        let bind_group = create_bind_group(self, &td, tag, &pipeline);
        let command = self
            .context
            .encode_command(workgroup_info, &pipeline, &bind_group);
        self.context.submit_command(command).ok().unwrap();

        td
    }

    fn map_broadcast<F: Fn(f32) -> f32 + Sync>(
        &self,
        out: &Self,
        _f: F,
        tag: &'static str,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        let workgroup_info = (&out.shape).into();
        let out_gpu_size = out.shape.gpu_byte_size();
        let output_buffer = out.context.create_output_buffer(
            out_gpu_size,
            tag,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let pipeline = self.context.get_or_create_pipeline(tag, workgroup_info)?;
        let bind_group = create_bind_group(self, out, tag, &pipeline);
        let command = self
            .context
            .encode_command(workgroup_info, &pipeline, &bind_group);
        self.context.submit_command(command).ok()?;
        Some(out.with_buffer(output_buffer))
    }

    fn zip<F: Fn(f32, f32) -> f32 + Sync>(
        &self,
        _other: &Self,
        _f: F,
        _tag: &'static str,
    ) -> Option<Self>
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
    a: &GpuTensorData<'_>,
    b: &GpuTensorData<'_>,
    operation: &str,
    pipeline: &ComputePipeline,
) -> BindGroup {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let b_gpu_size = b.shape.gpu_byte_size();
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
                    .context
                    .create_output_buffer(
                        b_gpu_size,
                        operation,
                        BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    )
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

#[cfg(test)]
mod tests {
    use crate::{shaping::shape::Shape, wgpu::wgpu_context::get_wgpu_context};

    use super::*;

    #[test]
    fn gpu_map_broadcast_test() {
        let shape = Shape::new(vec![3]);
        let strides = (&shape).into();
        let td = GpuTensorData::new(&[1., 2., 3.], shape, strides, get_wgpu_context());
        let res = td.map_broadcast(&td, |f| -f, "neg");
        let cpu_data = res.unwrap().to_cpu();
        println!("{:?}", cpu_data);
        assert!(true);
    }
}
