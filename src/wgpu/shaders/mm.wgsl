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
    return metadata[i + PREAMBLE + metadata[0u]];
}

fn b_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0u] * 2u];
}

fn b_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0u] * 2u + metadata[1u]];
}

fn out_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0u] * 2u + metadata[1u] * 2u];
}

fn out_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0u] * 2u + metadata[1u] * 2u + metadata[2u]];
}

fn a_tile_stride() -> u32 {
    if (a_shape(0u) > 1u) {
        return a_strides(0u);
    } else {
        return 0u;
    }
}

fn b_tile_stride() -> u32 {
    if (b_shape(0u) > 1u) {
        return b_strides(0u);
    } else {
        return 0u;
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
  return (a + b - 1u) / b;
}

@compute
@workgroup_size(1)
fn call(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let tx = workgroup_id.x * TILE_SIZE;
    let ty = workgroup_id.y * TILE_SIZE;
    let z = workgroup_id.z;

    let a_shape_len = metadata[0];
    let b_shape_len = metadata[1];
    let out_shape_len = metadata[2];

    let a_tile_stride = a_tile_stride();
    let b_tile_stride = b_tile_stride();

    var acc = 0.;
    let tiles = div_ceil(a_shape(a_shape_len - 1u), TILE_SIZE);
    for (var tile_index = 0u; tile_index < tiles; tile_index = tile_index + 1u) {
        let index = tile_index * TILE_SIZE;

        if ((lx + index) < a_shape(a_shape_len - 1u) &&
            (ly + ty) < a_shape(a_shape_len - 2u)) {
            let a_idx = z * a_tile_stride +
                (lx + index) * a_strides(a_shape_len - 1u) +
                (ly + ty) * a_strides(a_shape_len - 2u);
            a_tile[ly][lx] = input_a[a_idx];
        } else {
            a_tile[ly][lx] = 0.;
        }
    }

    // assume square matrix of size
    let size = 4u;

    let idx = ly * size + lx;

    if (lx < size && ly < size) {
        a_tile[ly][lx] = input_a[idx];
        b_tile[ly][lx] = input_b[idx];
    } else {
        a_tile[ly][lx] = 0.;
        b_tile[ly][lx] = 0.;
    }

    workgroupBarrier();

    if (ly < size && lx < size) {
        var acc = 0.;
        for (var i = 0u; i < size; i = i + 1u) {
            acc = fma(a_tile[ly][i], b_tile[i][lx], acc);
        }
        output[idx] = acc;
    }
}
