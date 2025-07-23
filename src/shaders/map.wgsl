@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<storage, read> shape: array<u32>;
@group(0) @binding(3)
var<storage, read> strides: array<u32>;

fn prod(start: u32, shape_len: u32) -> u32 {
    var result: u32 = 1u;
    for (var i = start; i < shape_len; i = i + 1u) {
        result *= shape[i];
    }
    return result;
}
