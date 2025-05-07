## allumette

`allumette` is a toy tensor library built for fun to better understand autodifferentiation.

It is based on [minitorch](minitorch.github.io).


### Usage

[Dataset](./src/training/dataset.rs) provides a few ways to create synthetic datasets.

```rust
let pts = 10;
let dataset = Dataset::simple(pts);
let hidden_layer_size = 3;
let learning_rate = 0.5;
let iterations = 200;
train::train(dataset, learning_rate, iterations, hidden_layer_size);
```

### Next up

- [ ] parallel backend
- [ ] gpu backend
- [ ] optimizations
