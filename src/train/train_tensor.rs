use crate::{
    data::dataset::Dataset,
    optim::optimizer::Optimizer,
    shaping::{shape::Shape, shaped::Shaped},
    tensor::{tensor::Tensor, tensor_data::TensorData},
};

use super::network::Network;

pub fn train(data: Dataset, learning_rate: f64, max_epochs: usize, hidden_layer_size: usize) {
    let mut network = Network::new(hidden_layer_size);
    let lr_tensor = Tensor::scalar(learning_rate);

    let x_shape = Shape::new(vec![data.x.len(), 2]);
    let x_strides = (&x_shape).into();
    let x_data = TensorData::new(flatten(&data.x), x_shape, x_strides);
    let x = Tensor::from_data(x_data);
    let y_data = TensorData::vec(data.y.iter().map(|u| *u as f64).collect());
    let y = Tensor::from_data(y_data);
    let n_shape = Shape::new(vec![data.n]);
    let one_shape = Shape::scalar(1);

    for epoch in 1..max_epochs + 1 {
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

        if epoch % 10 == 0 || epoch == max_epochs {
            let y2 = y.clone();
            let correct = out
                .clone()
                .gt(Tensor::scalar(0.5))
                .eq(y2)
                .sum(None)
                .item()
                .unwrap();
            println!("Epoch {epoch}, loss: {total_loss}, correct: {correct}\n");
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
