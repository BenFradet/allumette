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

// used to create local arrays
const MAX_DIMS: u32 = 32u;

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

fn add(a: f32, b: f32) -> f32 {
    return a + b;
}

fn mul(a: f32, b: f32) -> f32 {
    return a * b;
}

fn lt(a: f32, b: f32) -> f32 {
    if (a < b) {
        return 1.;
    } else {
        return 0.;
    }
}

fn eq(a: f32, b: f32) -> f32 {
    if (a == b) {
        return 1.;
    } else {
        return 0.;
    }
}

fn is_close(a: f32, b: f32) -> f32 {
    if (abs(a - b) < 0.001) {
        return 1.;
    } else {
        return 0.;
    }
}

fn ln_diff(a: f32, b: f32) -> f32 {
    if (a == 0.) {
        return b;
    } else {
        return b / a;
    }
}

fn relu_diff(a: f32, b: f32) -> f32 {
    if (a > 0.) {
        return b;
    } else {
        return 0.;
    }
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

fn broadcast_index_a(
    a_shape_len: u32,
    out_shape_len: u32,
    a_index: ptr<function, array<u32, MAX_DIMS>>,
    out_index: array<u32, MAX_DIMS>,
) {
    for (var i = 0u; i < a_shape_len; i = i + 1u) {
        if (a_shape(i) > 1u) {
            let idx = out_shape_len - a_shape_len + i;
            (*a_index)[i] = out_index[idx];
        } else {
            (*a_index)[i] = 0u;
        }
    }
}

fn broadcast_index_b(
    b_shape_len: u32,
    out_shape_len: u32,
    b_index: ptr<function, array<u32, MAX_DIMS>>,
    out_index: array<u32, MAX_DIMS>,
) {
    for (var i = 0u; i < b_shape_len; i = i + 1u) {
        if (b_shape(i) > 1u) {
            let idx = out_shape_len - b_shape_len + i;
            (*b_index)[i] = out_index[idx];
        } else {
            (*b_index)[i] = 0u;
        }
    }
}

fn index_to_position_a(
    len: u32,
    a_index: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += a_index[i] * a_strides(i);
    }
    return result;
}

fn index_to_position_b(
    len: u32,
    b_index: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += b_index[i] * b_strides(i);
    }
    return result;
}

fn index_to_position_out(
    len: u32,
    out_index: array<u32, MAX_DIMS>,
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += out_index[i] * out_strides(i);
    }
    return result;
}

@compute
@workgroup_size(1)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i > arrayLength(&output)) {
        return;
    }

    var a_index: array<u32, MAX_DIMS>;
    var b_index: array<u32, MAX_DIMS>;
    var out_index: array<u32, MAX_DIMS>;

    let a_shape_len = metadata[0u];
    let b_shape_len = metadata[1u];
    let out_shape_len = metadata[2u];

    to_index(i, out_shape_len, &out_index);
    broadcast_index_a(a_shape_len, out_shape_len, &a_index, out_index);
    broadcast_index_b(b_shape_len, out_shape_len, &b_index, out_index);

    let a_pos = index_to_position_a(a_shape_len, a_index);
    let b_pos = index_to_position_b(b_shape_len, b_index);
    let out_pos = index_to_position_out(out_shape_len, out_index);

    output[out_pos] = replace_with_operation(input_a[a_pos], input_b[b_pos]);
}
