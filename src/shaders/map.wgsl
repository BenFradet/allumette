@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<storage, read> shape: array<u32>;
@group(0) @binding(3)
var<storage, read> strides: array<u32>;

@group(0) @binding(4)
var<storage, read_write> index: array<u32>;

fn prod(start: u32, shape_len: u32) -> u32 {
    var result: u32 = 1u;
    for (var i = start; i < shape_len; i = i + 1u) {
        result *= shape[i];
    }
    return result;
}

fn to_index(ordinal: u32, shape_len: u32) {
    var remaining = ordinal;

    for (var i = 0u; i < shape_len; i = i + 1u) {
        let product = prod(i, shape_len);
        let divisor = product / shape[i];
        let index = remaining / divisor;
        remaining -= index * divisor;

        index[i] = index;
    }
}
