use crate::{
    backend::{backend::TensorBackend, backend_type::TensorBackendType},
    data::tensor_data::TensorData,
    optim::optimizer::Optimizer,
    shaping::shape::Shape,
    tensor::Tensor,
};

use super::{dataset::Dataset, network::Network};

pub fn train<BT: TensorBackendType, T: TensorBackend<BT>>(
    data: Dataset,
    learning_rate: f64,
    iterations: usize,
    hidden_layer_size: usize,
) {
    let mut network: Network<'_, BT, T> = Network::new(hidden_layer_size);
    let lr_tensor = Tensor::scalar(learning_rate);

    let x_shape = Shape::new(vec![data.x.len(), 2]);
    let x_strides = (&x_shape).into();
    let x_data = <T as TensorData>::from(flatten(&data.x), x_shape, x_strides);
    let x = Tensor::from_data(x_data);
    let y_data = <T as TensorData>::vec(data.y.iter().map(|u| *u as f64).collect());
    let y = Tensor::from_data(y_data);
    let n_shape = Shape::new(vec![data.n]);
    let one_shape = Shape::scalar(1);

    for iteration in 1..iterations + 1 {
        network.zero();

        let out = network.forward(x.clone()).view(&n_shape);
        let prob = (out.clone() * y.clone())
            + (out.clone() - Tensor::scalar(1.)) * (y.clone() - Tensor::scalar(1.));

        let loss = -prob.clone().ln();

        let res = (loss.clone() / Tensor::scalar(data.n as f64))
            .sum(None)
            .view(&one_shape)
            .backward();
        network.update(&res);

        let total_loss = loss.clone().sum(None).view(&one_shape).item().unwrap_or(0.);

        network.step(lr_tensor.clone());

        if iteration % 10 == 0 || iteration == iterations {
            let y2 = y.clone();
            let correct = out
                .clone()
                .gt(Tensor::scalar(0.5))
                .eq(y2)
                .sum(None)
                .item()
                .unwrap();
            println!("Iteration {iteration}, loss: {total_loss}, correct: {correct}\n");
        }
    }
}

fn flatten(d: &[(f64, f64)]) -> Vec<f64> {
    d.iter()
        .fold(Vec::with_capacity(d.len() * 2), |mut acc, t| {
            acc.push(t.0);
            acc.push(t.1);
            acc
        })
}
