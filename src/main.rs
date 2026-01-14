use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend, GpuBackend},
    training::{
        dataset::Dataset,
        debugger::{ChattyDebugger, TerseDebugger},
        train,
    },
};

fn main() {
    let pts = 1000;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.1;
    let iterations = 200;
    train::train::<GpuBackend, TerseDebugger>(
        dataset,
        learning_rate,
        iterations,
        hidden_layer_size,
    );
}
