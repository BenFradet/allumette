use allumette::{
    backend::backend_type::{Par, Seq},
    data::cpu_tensor_data::CpuTensorData,
    training::{dataset::Dataset, train},
};

fn main() {
    let pts = 1000;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.5;
    let iterations = 500;
    train::train::<Par, CpuTensorData>(dataset, learning_rate, iterations, hidden_layer_size);
}
