## allumette

`allumette` is a toy tensor library built for fun to better understand autodifferentiation.

It is inspired by a small cohort of projects:
- [minitorch](minitorch.github.io)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [burn](https://github.com/tracel-ai/burn)
- [candle](https://github.com/huggingface/candle)
- [tensorken](https://github.com/kurtschelfthout/tensorken)


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
- [x] gpu backend
- [ ] visualization
- [ ] ergonomics (associated types)
- [ ] convolution
- [ ] optimizations
- [ ] tensor dimension as const generic

### Gotchas / learnings

#### proptest

Seems like `proptest` distributions are truly uniform unlike quickcheck or scalacheck which do
hotspot values.
`relu'(x)` is undefined when `x = 0` and by convention I had chosen 0. The central diff however
reports nonsensical values.
The bug was there for months until I ported the same logic to GPU where I hit on 0 by chance.
C.f. https://github.com/proptest-rs/proptest/issues/82

#### proptest & GPU

GPU is fast except going to and from the CPU which happens a lot with prop tests

#### IGPs are slow
