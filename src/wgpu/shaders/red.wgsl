@group(0) @binding(0)
var<storage, read> input: array<f32>;

// only 4 storage buffers allowed
// structure:
// a shape len
// out shape len
// index of the dimension along which to reduce
// a shape / a strides
// out shape / out strides
@group(0) @binding(1)
var<storage, read> metadata: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 1u;
// lives in the wg address space, hence shared between invcations of a wg
var<workgroup> shared_block: array<f32, WG_SIZE>;

// used to create local arrays
const MAX_DIMS: u32 = 32u;

// shape lengths
const PREAMBLE: u32 = 3u;

fn a_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE];
}

fn a_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0]];
}

fn out_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u];
}

fn out_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u + metadata[1]];
}

fn prod(
    start: u32,
    shape_len: u32,
) -> u32 {
    var result: u32 = 1u;
    for (var i = start; i < shape_len; i = i + 1u) {
        result *= out_shape(i);
    }
    return result;
}

fn to_index(
    ordinal: u32,
    shape_len: u32,
    out_index: ptr<function, array<u32, MAX_DIMS>>,
) {
    var remaining = ordinal;
    for (var i = 0u; i < shape_len; i = i + 1u) {
        let product = prod(i, shape_len);
        let divisor = product / out_shape(i);
        let index = remaining / divisor;
        remaining -= index * divisor;

        (*out_index)[i] = index;
    }
}

fn index_to_position_a(
    shape_len: u32,
    index: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < shape_len; i = i + 1u) {
        result += index[i] * a_strides(i);
    }
    return result;
}

fn index_to_position_out(
    shape_len: u32,
    index: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < shape_len; i = i + 1u) {
        result += index[i] * out_strides(i);
    }
    return result;
}

fn sum(a: f32, b: f32) -> f32 {
    return a + b;
}

fn all(a: f32, b: f32) -> f32 {
  return a * b;
}

@compute
@workgroup_size(WG_SIZE)
fn call(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_nb_elems = arrayLength(&output);
    // active workgroup flag to avoid early returns
    let wg_active = workgroup_id.x < output_nb_elems;

    var a_index: array<u32, MAX_DIMS>;
    var out_index: array<u32, MAX_DIMS>;

    let a_shape_len = metadata[0];
    let out_shape_len = metadata[1];
    let reduce_dim = metadata[2];
    let reduce_default = f32(replace_with_default);
    let reduce_size = a_shape(reduce_dim);

    var out_pos: u32 = 0u;
    if (wg_active) {
        to_index(workgroup_id.x, out_shape_len, &out_index);
        out_pos = index_to_position_out(out_shape_len, out_index);
    }

    if (wg_active && local_id.x < reduce_size) {
        out_index[reduce_dim] = local_id.x;
        let pos = index_to_position_a(a_shape_len, out_index);
        shared_block[local_id.x] = input[pos];
    }

    // all invocations must reach the barrier
    // otherwise it is undefined behaviour
    // no early returns allowed
    workgroupBarrier();

    if (wg_active && local_id.x == 0u) {
        var acc = reduce_default;
        for (var i = 0u; i < reduce_size; i = i + 1u) {
            acc = replace_with_operation(acc, shared_block[i]);
        }
        output[out_pos] = acc;
    }
}
