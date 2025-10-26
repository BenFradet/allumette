@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// only 4 storage buffers allowed
// structure:
// in shape len
// out shape len
// in shape / in strides / in index
// out shape / out strides / out index
@group(0) @binding(2)
var<storage, read_write> metadata: array<u32>;

// ndims
const PREAMBLE: u32 = 2u;

fn in_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE];
}

fn in_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0]];
}

fn in_index(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 2u];
}

fn out_shape(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 3u];
}

fn out_strides(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 3u + metadata[1]];
}

fn out_index(i: u32) -> u32 {
    return metadata[i + PREAMBLE + metadata[0] * 3u + metadata[1] * 2u];
}

fn id(in: f32) -> f32 {
    return in;
}

fn neg(in: f32) -> f32 {
    return -in;
}

fn inv(in: f32) -> f32 {
    if (in == 0.0) {
        return 0.0;
    }
    return 1.0 / in;
}

fn relu(in: f32) -> f32 {
    return max(0.0, in);
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
) {
    var remaining = ordinal;
    for (var i = 0u; i < shape_len; i = i + 1u) {
        let product = prod(i, shape_len);
        let divisor = product / out_shape(i);
        let index = remaining / divisor;
        remaining -= index * divisor;

        metadata[i + PREAMBLE + metadata[0] * 3u + metadata[1] * 2u] = index;
        //out_index[i] = index;
    }
}

fn broadcast_index(
    in_shape_len: u32,
    out_shape_len: u32,
) {
    for (var i = 0u; i < in_shape_len; i = i + 1u) {
        let ii = i + PREAMBLE + metadata[0] * 2u;
        if (in_shape(i) > 1u) {
            let idx = out_shape_len - in_shape_len + i;
            //in_index[i] = out_index[idx];
            metadata[ii] = out_index(idx);
        } else {
            //in_index[i] = 0u;
            metadata[ii] = 0u;
        }
    }
}

// haven't found a way not to copy/paste given array<u32, N> needs constant indexing
fn index_to_position_in(
    len: u32
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += in_index(i) * in_strides(i);
    }
    return result;
}

fn index_to_position_out(
    len: u32
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += out_index(i) * out_strides(i);
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

    let in_shape_len = metadata[0];
    let out_shape_len = metadata[1];
    to_index(i, out_shape_len);
    broadcast_index(in_shape_len, out_shape_len);
    let in_pos = index_to_position_in(in_shape_len);
    let out_pos = index_to_position_out(out_shape_len);
    output[out_pos] = replace_with_actual_operation(input[in_pos]);
    //output[i] = replace_with_actual_operation(input[i]);
}