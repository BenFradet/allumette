use allumette::{
    backend::backend_type::{Gpu, Seq},
    data::{cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData},
    training::{dataset::Dataset, train},
};

fn main() {
    let pts = 100;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.01;
    let iterations = 500;
    train::train::<f32, Gpu, GpuTensorData>(dataset, learning_rate, iterations, hidden_layer_size);
}
