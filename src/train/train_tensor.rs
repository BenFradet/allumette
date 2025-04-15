use crate::{
    data::dataset::Dataset,
    optim::{optimizer::Optimizer, sgd::SGD},
    tensor::{shaping::shape::Shape, tensor::Tensor, tensor_data::TensorData},
};

use super::network::Network;

pub fn train(data: Dataset, learning_rate: f64, max_epochs: usize, hidden_layer_size: usize) -> () {
    let network = Network::new(hidden_layer_size);
    let mut tensors = network.init();
    let optim = SGD::new(learning_rate);
    let mut losses = vec![];

    let x_shape = Shape::new(vec![data.x.len(), 2]);
    let x_strides = (&x_shape).into();
    let x_data = TensorData::new(flatten(&data.x), x_shape, x_strides);
    let x = Tensor::from_data(x_data);
    let y_data = TensorData::vec(data.y.iter().map(|u| *u as f64).collect()).unwrap();
    let y = Tensor::from_data(y_data);
    let n_tensor = Tensor::scalar(data.n as f64);
    let n_shape = Shape::new(vec![data.n]);
    let one_shape = Shape::scalar(1);

    for epoch in 1..max_epochs + 1 {
        tensors = optim.zero_grad(tensors);

        let out = network.forward(x.clone(), &tensors).view(&n_shape).unwrap();
        let prob = (out.clone() * y.clone())
            + (out.clone() - Tensor::scalar(1.)) * (y.clone() - Tensor::scalar(1.));

        let loss = -prob.ln();
        let res = (loss.clone() / n_tensor.clone())
            .sum(None)
            .view(&one_shape)
            .unwrap()
            .backward();
        tensors.extend(res);
        let total_loss = loss
            .sum(None)
            .view(&one_shape)
            .unwrap()
            .item()
            .unwrap_or(0.);
        losses.push(total_loss);

        tensors = optim.update(tensors);

        if epoch % 10 == 0 || epoch == max_epochs {
            let y2 = y.clone();
            let correct = out
                .clone()
                .gt(Tensor::scalar(0.5))
                .eq(y2)
                .sum(None)
                .item()
                .unwrap();
            println!("Epoch {epoch}, loss: {total_loss}, correct: {correct}");
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
