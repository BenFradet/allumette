use allumette::{
    backend::backend_type::{Gpu, Seq},
    data::{cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData},
    training::{dataset::Dataset, train},
};

fn main() {
    let pts = 20;
    //let dataset = Dataset::simple(pts);
    let dataset = Dataset {
        n: pts,
        x: vec![
            (0.41133654, 0.45256868),
            (0.810811, 0.3776284),
            (0.7816291, 0.2726979),
            (0.47309753, 0.3503597),
            (0.81114423, 0.78229),
            (0.53246886, 0.43584824),
            (0.49532536, 0.008408654),
            (0.9767621, 0.032501426),
            (0.85100466, 0.8881119),
            (0.020138174, 0.97354126),
            (0.8623625, 0.42854437),
            (0.76391965, 0.5381634),
            (0.96961766, 0.47776037),
            (0.06688703, 0.8401424),
            (0.24678175, 0.42118937),
            (0.11337148, 0.83761895),
            (0.26896992, 0.3988581),
            (0.5591019, 0.2677568),
            (0.070258304, 0.248804),
            (0.32880816, 0.542155),
        ],
        y: vec![1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
    };
    let hidden_layer_size = 3;
    let learning_rate = 0.05;
    let iterations = 1;
    //cpu layer 1 grad [0.0, -0.00046193235971115645, 0.007418926796722089, 0.0, -0.0016358296782338297, -0.00019212343075844513]
    //gpu layer 1 grad [0.0, 0.0, 0.0074904417, 0.0, 0.0, 0.0008900426]
    train::train::<f32, Gpu, GpuTensorData>(dataset, learning_rate, iterations, hidden_layer_size);
}
