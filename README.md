## allumette

`allumette` is a toy tensor library built for fun to better understand autodifferentiation.

It is based on [minitorch](minitorch.github.io).


### Usage

[Dataset](./src/training/dataset.rs) provides a few ways to create synthetic datasets.

```rust
use allumette::{
    backend::backend_type::{Par, Seq},
    data::cpu_tensor_data::CpuTensorData,
    training::{dataset::Dataset, train},
};

let pts = 10;
let dataset = Dataset::simple(pts);
let hidden_layer_size = 3;
let learning_rate = 0.5;
let iterations = 200;
// use Par instead of Seq to leverage rayon's parallel iterators
train::train::<Seq, CpuTensorData>(dataset, learning_rate, iterations, hidden_layer_size);
```

### Build and dependencies

Part of the codebase makes use of the `generic_const_exprs` and `trait_alias` experimental features
so it requires nightly.

The set of dependencies is otherwise pretty limited:
- `wgpu` for the GPU runtime
- `rayon` for the parallel CPU runtime
- `flume` and `futures` for wgpu callbacks
- `bytemuck` to convert binary buffers copied to/from the GPU
- `proptest` for property-based testing
- `rand` for synthetic data generation

### Next up

- [x] parallel backend
- [ ] gpu backend
- [ ] ergonomics
- [ ] optimizations
- [ ] make encoding tensor dimension as a const generic work
