@group(0) @binding(0)
var<storage, read> input_a: array<f32>;
@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

// only 4 storage buffers allowed
// structure:
// a shape len
// b shape len
// out shape len
// a shape / a strides
// b shape / b strides
// out shape / out strides
@group(0) @binding(2)
var<storage, read> metadata: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

var<workgroup> a_tile: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> b_tile: array<array<f32, TILE_SIZE>, TILE_SIZE>;

// used to create local arrays
const TILE_SIZE: u32 = 32u;

// shape lengths
const PREAMBLE: u32 = 3u;

fn a_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE];
}

fn a_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0]];
}

fn b_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u];
}

fn b_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u + metadata[1]];
}

fn out_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u + metadata[1] * 2u];
}

fn out_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u + metadata[1] * 2u + metadata[2]];
}

@compute
@workgroup_size(1)
fn call(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
}
