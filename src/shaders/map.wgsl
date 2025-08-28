@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<storage, read> in_shape: array<u32>;
@group(0) @binding(3)
var<storage, read> out_shape: array<u32>;

@group(0) @binding(4)
var<storage, read> in_strides: array<u32>;
@group(0) @binding(5)
var<storage, read> out_strides: array<u32>;

const MAX_DIMS: u32 = 32u;

fn prod(
    start: u32,
    shape_len: u32,
    shape: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 1u;
    for (var i = start; i < shape_len; i = i + 1u) {
        result *= shape[i];
    }
    return result;
}

fn to_index(
    ordinal: u32,
    shape_len: u32,
    shape: array<u32, MAX_DIMS>,
) -> array<u32, MAX_DIMS> {
    var remaining = ordinal;
    var out_index: array<u32, MAX_DIMS>;

    for (var i = 0u; i < shape_len; i = i + 1u) {
        let product = prod(i, shape_len, shape);
        let divisor = product / shape[i];
        let index = remaining / divisor;
        remaining -= index * divisor;

        out_index[i] = index;
    }
    return out_index;
}

fn broadcast_index(
    in_index: array<u32, MAX_DIMS>,
    in_shape_len: u32,
    in_shape: array<u32, MAX_DIMS>,
    out_shape_len: u32,
    out_shape: array<u32, MAX_DIMS>,
) -> array<u32, MAX_DIMS> {
    var out_index: array<u32, MAX_DIMS>;

    for (var i = 0u; i < out_shape_len; i = i + 1u) {
        if out_shape[i] > 1u {
            out_index[i] = in_index[in_shape_len - out_shape_len - i];
        } else {
            out_index[i] = 0u;
        }
    }

    return out_index;
}

fn index_to_position(
    index: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    len: u32
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += index[i] * strides[i];
    }
    return result;
}

// parlor trick taken from https://github.com/kurtschelfthout/tensorken
fn replace_with_actual_operation(in: f32) -> f32 { discard; }

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if i > arrayLength(&output) {
        return;
    }

    var out_index: array<u32, MAX_DIMS>;
    var in_index: array<u32, MAX_DIMS>;
    let out_shape_len = arrayLength(&out_shape);
    let in_shape_len = arrayLength(&in_shape);
    to_index(i, shape_len, out_index);
    broadcast_index(out_index, in_index, out_shape_len, in_shape_len);
    let in_pos = index_to_position(in_index, in_strides);
    let out_pos = index_to_position(out_index, out_strides);
    output[out_pos] = replace_with_actual_operation(input[in_pos]);
}