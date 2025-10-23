use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferUsages,
    ComputePipeline,
};

use crate::{
    backend::{backend::TensorBackend, backend_type::Gpu},
    data::gpu_tensor_data::GpuTensorData,
};

impl TensorBackend<f32, Gpu> for GpuTensorData<'_> {
    // TODO: rm unwrap
    fn map<F: Fn(f32) -> f32 + Sync>(&self, _f: F, tag: &'static str) -> Self {
        todo!();
        //let shape = self.shape.clone();
        //let gpu_size = shape.gpu_byte_size();
        //// make contiguous
        //let strides = (&shape).into();

        //let buffer = self.context.create_output_buffer(
        //    gpu_size,
        //    tag,
        //    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        //);
        //let td = GpuTensorData::from_buffer(shape, strides, buffer, self.context);

        //let workgroup_info = (&td.shape).into();
        //let pipeline = td
        //    .context
        //    .get_or_create_pipeline(tag, &workgroup_info)
        //    .unwrap();

        //let bind_group = create_bind_group(self, &td, tag, &pipeline);
        //let command = self
        //    .context
        //    .encode_command(&workgroup_info, &pipeline, &bind_group);
        //self.context.submit_command(command).ok().unwrap();

        //td
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
        // TODO: use out.buffer.size?
        let out_gpu_size = out.shape.gpu_byte_size();
        let output_buffer = out.context.create_output_buffer(
            out_gpu_size,
            tag,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let pipeline = self.context.get_or_create_pipeline(tag, &workgroup_info)?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let metadata_buffer = create_metadata_buffer(self, out);
        let bind_group = self.context.create_bind_group(
            &self.buffer,
            &output_buffer,
            &metadata_buffer,
            &bind_group_layout,
        );
        let command = self
            .context
            .encode_command(&workgroup_info, &pipeline, &bind_group);
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

fn create_metadata_buffer(a: &GpuTensorData<'_>, b: &GpuTensorData<'_>) -> Buffer {
    let mut contents = Vec::with_capacity(2 + 3 * a.shape.len() + 3 * b.shape.len());
    contents.push(a.shape.len());
    contents.push(b.shape.len());
    contents.extend(a.shape.data());
    contents.extend(a.strides.data());
    contents.extend(a.shape.idx(0).data());
    contents.extend(b.shape.data());
    contents.extend(b.strides.data());
    contents.extend(b.shape.idx(0).data());

    let metadata: Vec<u32> = contents
        .iter()
        .map(|u| u32::try_from(*u).unwrap())
        .collect();

    println!("metadata: {metadata:?}");

    a.device().create_buffer_init(&BufferInitDescriptor {
        label: Some("meta"),
        contents: bytemuck::cast_slice(&metadata),
        usage: BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    use crate::{shaping::shape::Shape, wgpu::wgpu_context::get_wgpu_context};

    use super::*;

    #[test]
    fn gpu_map_broadcast_test() {
        let shape = Shape::new(vec![2, 3]);
        let strides = (&shape).into();
        let td = GpuTensorData::new(&[1.; 6], shape, strides, get_wgpu_context());
        let res = td.map_broadcast(&td, |f| -f, "neg");
        let cpu_data = res.unwrap().to_cpu();
        println!("{:?}", cpu_data);
        assert!(true);
    }
}
