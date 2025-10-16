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

@group(0) @binding(6)
var<storage, read_write> in_index: array<u32>;
@group(0) @binding(7)
var<storage, read_write> out_index: array<u32>;

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
        result *= out_shape[i];
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
        let divisor = product / out_shape[i];
        let index = remaining / divisor;
        remaining -= index * divisor;

        out_index[i] = index;
    }
}

fn broadcast_index(
    in_shape_len: u32,
    out_shape_len: u32,
) {
    for (var i = 0u; i < in_shape_len; i = i + 1u) {
        if (in_shape[i] > 1u) {
            let idx = out_shape_len - in_shape_len - i;
            in_index[i] = out_index[idx];
        } else {
            in_index[i] = 0u;
        }
    }
}

// haven't found a way not to copy/paste given array<u32, N> needs constant indexing
fn index_to_position_in(
    len: u32
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += in_index[i] * in_strides[i];
    }
    return result;
}

fn index_to_position_out(
    len: u32
) -> u32 {
    var result: u32 = 0u;
    for (var i = 0u; i < len; i = i + 1u) {
        result += out_index[i] * out_strides[i];
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

    //let out_shape_len = arrayLength(&out_shape);
    //let in_shape_len = arrayLength(&in_shape);
    //to_index(i, out_shape_len);
    //broadcast_index(in_shape_len, out_shape_len);
    //let in_pos = index_to_position_in(in_shape_len);
    //let out_pos = index_to_position_out(out_shape_len);
    output[i] = replace_with_actual_operation(input[i]);
}