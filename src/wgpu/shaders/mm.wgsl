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
// might want to dynamically change it
const TILE_SIZE: u32 = 16u;

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

fn a_fst_s() -> u32 {
    if (a_shape(0u) > 1u) {
        return a_strides(0u);
    } else {
        return 0u;
    }
}

fn b_fst_s() -> u32 {
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
    // positions of the thread in the tile
    let lx = local_id.x;
    let ly = local_id.y;
    // positions of the tile in the grid
    let tx = workgroup_id.x * TILE_SIZE;
    let ty = workgroup_id.y * TILE_SIZE;
    // batch index
    let z = workgroup_id.z;

    let a_shape_len = metadata[0];
    let b_shape_len = metadata[1];
    let out_shape_len = metadata[2];

    let a_col = a_shape(a_shape_len - 1u);
    let a_row = a_shape(a_shape_len - 2u);
    let a_fst_s = a_fst_s();
    let a_col_s = a_strides(a_shape_len - 1u);
    let a_row_s = a_strides(a_shape_len - 2u);

    let b_col = b_shape(b_shape_len - 1u);
    let b_row = b_shape(b_shape_len - 2u);
    let b_fst_s = b_fst_s();
    let b_col_s = b_strides(b_shape_len - 1u);
    let b_row_s = b_strides(b_shape_len - 2u);

    var acc = 0.;
    let n_tiles = div_ceil(a_col, TILE_SIZE);
    for (var tile = 0u; tile < n_tiles; tile = tile + 1u) {
        // starting index of the tile
        let tile_idx = tile * TILE_SIZE;

        // each thread loads one element of a_tile
        if ((lx + tile_idx) < a_col && (ly + ty) < a_row) {
            let a_idx = z * a_fst_s + (lx + tile_idx) * a_col_s + (ly + ty) * a_row_s;
            a_tile[ly][lx] = input_a[a_idx];
        } else {
            a_tile[ly][lx] = 0.;
        }

        // each thread loads one element of b_tile
        if ((lx + tx) < b_col && (ly + tile_idx) < b_row) {
            let b_idx = z * b_fst_s + (lx + tx) * b_col_s + (ly + tile_idx) * b_row_s;
            b_tile[ly][lx] = input_b[b_idx];
        } else {
            b_tile[ly][lx] = 0.;
        }

        workgroupBarrier();

        // dot product of a row and b col (look into builtin op for dotp)
        for (var inner_idx = 0u; inner_idx < TILE_SIZE; inner_idx = inner_idx + 1u) {
            acc = fma(a_tile[ly][inner_idx], b_tile[inner_idx][lx], acc);
        }
    }

    let out_col = out_shape(out_shape_len - 1u);
    let out_row = out_shape(out_shape_len - 2u);
    let out_fst_s = out_strides(0u);
    let out_col_s = out_strides(out_shape_len - 1u);
    let out_row_s = out_strides(out_shape_len - 2u);
    if ((lx + tx) < out_col && (ly + ty) < out_row) {
        let idx = z * out_fst_s + (lx + tx) * out_col_s + (ly + ty) * out_row_s;
        output[idx] = acc;
    }
}
