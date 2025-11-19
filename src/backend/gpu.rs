use wgpu::BufferUsages;

use crate::{
    backend::{backend::TensorBackend, backend_type::Gpu},
    data::gpu_tensor_data::GpuTensorData, shaping::shape::Shape,
};

// TODO: abstract fn, they're all doing mostly the same
impl TensorBackend<f32, Gpu> for GpuTensorData<'_> {
    // TODO: rm unwraps
    fn map<F: Fn(f32) -> f32 + Sync>(&self, _f: F, tag: &'static str) -> Self {
        let workgroup_info = (&self.shape).into();
        let gpu_size = self.shape.gpu_byte_size();

        let output_buffer = self.context.create_output_buffer(
            gpu_size,
            tag,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let pipeline = self
            .context
            .get_or_create_pipeline(tag, workgroup_info, 3)
            .unwrap();
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let metadata_buffer = self
            .context
            .create_metadata_buffer(&[&self.shape, &self.shape], &[]);
        let bind_group = self.context.create_bind_group(
            &[&self.buffer, &metadata_buffer, &output_buffer],
            &bind_group_layout,
        );
        let command = self
            .context
            .encode_command(&workgroup_info, &pipeline, &bind_group);
        self.context.submit_command(command).ok().unwrap();

        self.with_buffer(output_buffer)
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

        let pipeline = self
            .context
            .get_or_create_pipeline(tag, workgroup_info, 3)?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let metadata_buffer = self
            .context
            .create_metadata_buffer(&[&self.shape, &out.shape], &[]);
        let bind_group = self.context.create_bind_group(
            &[&self.buffer, &metadata_buffer, &output_buffer],
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
        other: &Self,
        _f: F,
        tag: &'static str,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        let shape = if self.shape == other.shape {
            self.shape.clone()
        } else {
            self.shape.broadcast(&other.shape)?
        };
        let workgroup_info = (&shape).into();
        let gpu_size = shape.gpu_byte_size();
        let output_buffer = self.context.create_output_buffer(
            gpu_size,
            tag,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let pipeline = self
            .context
            .get_or_create_pipeline(tag, workgroup_info, 4)?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let metadata_buffer =
            self.context
                .create_metadata_buffer(&[&self.shape, &other.shape, &shape], &[]);
        let bind_group = self.context.create_bind_group(
            &[
                &self.buffer,
                &other.buffer,
                &metadata_buffer,
                &output_buffer,
            ],
            &bind_group_layout,
        );

        let command = self
            .context
            .encode_command(&workgroup_info, &pipeline, &bind_group);
        self.context.submit_command(command).ok()?;

        let strides = (&shape).into();
        Some(Self::from_buffer(
            shape,
            strides,
            output_buffer,
            self.context,
        ))
    }

    fn reduce<F: Fn(f32, f32) -> f32 + Sync>(&self, _f: F, dim: usize, init: f32, tag: &'static str) -> Option<Self>
    where
        Self: Sized,
    {
        if dim < self.shape.data().len() {
            let mut shape_data = self.shape.data().to_vec();
            shape_data[dim] = 1;
            let shape = Shape::new(shape_data);

            let workgroup_info = (&shape).into();
            let gpu_size = shape.gpu_byte_size();
            let output_buffer = self.context.create_output_buffer(
                gpu_size,
                tag,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );

            let pipeline = self
                .context
                .get_or_create_pipeline(tag, workgroup_info, 3)?;
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let metadata_buffer =
                self.context
                    .create_metadata_buffer(&[&self.shape, &shape], &[dim]);
            let bind_group = self.context.create_bind_group(
                &[
                    &self.buffer,
                    &metadata_buffer,
                    &output_buffer,
                ],
                &bind_group_layout,
            );

            let command = self
                .context
                .encode_command(&workgroup_info, &pipeline, &bind_group);
            self.context.submit_command(command).ok()?;

            let strides = (&shape).into();
            Some(Self::from_buffer(
                shape,
                strides,
                output_buffer,
                self.context,
            ))
        } else {
            None
        }
    }

    fn matmul(&self, _other: &Self) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{shaping::shape::Shape, wgpu::wgpu_context::get_wgpu_context};

    use super::*;

    #[test]
    fn gpu_map_broadcast_test() {
        let shape = Shape::new(vec![2, 4]);
        let strides = (&shape).into();
        let input: Vec<_> = (1..9).map(|u| u as f32).collect();
        let td = GpuTensorData::new(&input, shape, strides, get_wgpu_context());
        let res = td.map_broadcast(&td, |f| -f, "neg");
        let cpu_data = res.unwrap().to_cpu();
        assert_eq!(vec![-1., -2., -3., -4., -5., -6., -7., -8.], cpu_data);
    }

    #[test]
    fn cpu_zip_test() {
        // shape lengths
        const PREAMBLE: usize = 3;

        fn a_shape(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE]
        }

        fn a_strides(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0]]
        }

        fn b_shape(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 2]
        }

        fn b_strides(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 2 + metadata[1]]
        }

        fn out_shape(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 2 + metadata[1] * 2]
        }

        fn out_strides(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 2 + metadata[1] * 2 + metadata[2]]
        }

        fn add(a: f32, b: f32) -> f32 {
            a + b
        }

        fn prod(metadata: &[usize], start: usize, shape_len: usize) -> usize {
            let mut result: usize = 1;
            for i in start..shape_len {
                result *= out_shape(metadata, i);
            }
            result
        }

        fn to_index(metadata: &[usize], ordinal: usize, shape_len: usize, out_index: &mut [usize]) {
            let mut remaining = ordinal;
            #[allow(clippy::needless_range_loop)]
            for i in 0..shape_len {
                let product = prod(metadata, i, shape_len);
                let divisor = product / out_shape(metadata, i);
                let index = remaining / divisor;
                remaining -= index * divisor;

                out_index[i] = index;
            }
        }

        fn broadcast_index_a(
            metadata: &[usize],
            a_shape_len: usize,
            out_shape_len: usize,
            a_index: &mut [usize],
            out_index: &[usize],
        ) {
            #[allow(clippy::needless_range_loop)]
            for i in 0..a_shape_len {
                if a_shape(metadata, i) > 1 {
                    let idx = out_shape_len - a_shape_len + i;
                    a_index[i] = out_index[idx];
                } else {
                    a_index[i] = 0;
                }
            }
        }

        fn broadcast_index_b(
            metadata: &[usize],
            b_shape_len: usize,
            out_shape_len: usize,
            b_index: &mut [usize],
            out_index: &[usize],
        ) {
            #[allow(clippy::needless_range_loop)]
            for i in 0..b_shape_len {
                if b_shape(metadata, i) > 1 {
                    let idx = out_shape_len - b_shape_len + i;
                    b_index[i] = out_index[idx];
                } else {
                    b_index[i] = 0;
                }
            }
        }

        fn index_to_position_a(metadata: &[usize], len: usize, a_index: &[usize]) -> usize {
            let mut result: usize = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..len {
                result += a_index[i] * a_strides(metadata, i);
            }
            result
        }

        fn index_to_position_b(metadata: &[usize], len: usize, b_index: &[usize]) -> usize {
            let mut result: usize = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..len {
                result += b_index[i] * b_strides(metadata, i);
            }
            result
        }

        fn index_to_position_out(metadata: &[usize], len: usize, out_index: &[usize]) -> usize {
            let mut result: usize = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..len {
                result += out_index[i] * out_strides(metadata, i);
            }
            result
        }

        let input_a = [1., 2., 3., 4., 5., 6., 7., 8.];
        let input_b = [100.];
        let mut output = [0.; 8];

        let mut a_index = [0; 32];
        let mut b_index = [0; 32];
        let mut out_index = [0; 32];

        let metadata = [
            4, 1, 4, 2, 2, 1, 2, 4, 2, 2, 1, 1, 1, 2, 2, 1, 2, 4, 2, 2, 1,
        ];

        let a_shape_len = metadata[0];
        let b_shape_len = metadata[1];
        let out_shape_len = metadata[2];

        for i in 0..output.len() {
            to_index(&metadata, i, out_shape_len, &mut out_index);
            broadcast_index_a(
                &metadata,
                a_shape_len,
                out_shape_len,
                &mut a_index,
                &out_index,
            );
            broadcast_index_b(
                &metadata,
                b_shape_len,
                out_shape_len,
                &mut b_index,
                &out_index,
            );

            let a_pos = index_to_position_a(&metadata, a_shape_len, &a_index);
            let b_pos = index_to_position_b(&metadata, b_shape_len, &b_index);
            let out_pos = index_to_position_out(&metadata, out_shape_len, &out_index);

            output[out_pos] = add(input_a[a_pos], input_b[b_pos]);
        }
        assert_eq!(vec![101., 102., 103., 104., 105., 106., 107., 108.], output);
    }

    #[test]
    fn cpu_map_broadcast_test() {
        const PREAMBLE: usize = 2;

        fn in_shape(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE]
        }

        fn in_strides(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0]]
        }

        fn in_index(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 2]
        }

        fn out_shape(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 3]
        }

        fn out_strides(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 3 + metadata[1]]
        }

        fn out_index(metadata: &[usize], i: usize) -> usize {
            metadata[i + PREAMBLE + metadata[0] * 3 + metadata[1] * 2]
        }

        fn prod(metadata: &[usize], start: usize, shape_len: usize) -> usize {
            let mut result = 1;
            for i in start..shape_len {
                result *= out_shape(metadata, i);
            }
            result
        }

        fn to_index(metadata: &mut [usize], ordinal: usize, shape_len: usize) {
            let mut remaining = ordinal;
            for i in 0..shape_len {
                let product = prod(metadata, i, shape_len);
                let divisor = product / out_shape(metadata, i);
                let index = remaining / divisor;
                remaining -= index * divisor;

                let idx = i + PREAMBLE + metadata[0] * 3 + metadata[1] * 2;
                metadata[idx] = index;
            }
        }

        fn broadcast_index(metadata: &mut [usize], in_shape_len: usize, out_shape_len: usize) {
            for i in 0..in_shape_len {
                let ii = i + PREAMBLE + metadata[0] * 2;
                if in_shape(metadata, i) > 1 {
                    let idx = out_shape_len - in_shape_len + i;
                    metadata[ii] = out_index(metadata, idx);
                } else {
                    metadata[ii] = 0;
                }
            }
        }

        fn index_to_position_in(metadata: &[usize], len: usize) -> usize {
            let mut result = 0;
            for i in 0..len {
                result += in_index(metadata, i) * in_strides(metadata, i);
            }
            result
        }

        fn index_to_position_out(metadata: &[usize], len: usize) -> usize {
            let mut result = 0;
            for i in 0..len {
                result += out_index(metadata, i) * out_strides(metadata, i);
            }
            result
        }

        let mut metadata = vec![2, 2, 2, 3, 3, 1, 0, 0, 2, 3, 3, 1, 0, 0];
        let in_shape_len = metadata[0];
        let out_shape_len = metadata[1];
        let input: Vec<isize> = (1..7).collect();
        let mut output: Vec<isize> = vec![0; 6];
        for i in 0..input.len() {
            to_index(&mut metadata, i, out_shape_len);
            broadcast_index(&mut metadata, in_shape_len, out_shape_len);
            let in_pos = index_to_position_in(&metadata, in_shape_len);
            let out_pos = index_to_position_out(&metadata, out_shape_len);
            output[out_pos] = -input[in_pos];
        }
        assert_eq!(vec![-1, -2, -3, -4, -5, -6], output);
    }
}
