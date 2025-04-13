use crate::{data::dataset::Dataset, optim::{optimizer::Optimizer, sgd::SGD}, tensor::{shaping::shape::Shape, tensor::Tensor, tensor_data::TensorData}};

use super::network::Network;

pub fn train(data: Dataset, learning_rate: f64, max_epochs: usize, hidden_layer_size: usize) -> () {
    let network = Network::new(hidden_layer_size);
    let mut tensors = network.init();
    let optim = SGD::new(learning_rate);
    //let mut losses = vec![];

    let x_shape = Shape::new(vec![data.x.len(), 2]);
    let x_strides = (&x_shape).into();
    let x_data = TensorData::new(flatten(&data.x), x_shape, x_strides);
    let x = Tensor::from_data(x_data);
    let y_data = TensorData::vec(data.y.iter().map(|u| *u as f64).collect()).unwrap();
    let y = Tensor::from_data(y_data);
    let n_shape = Shape::new(vec![data.n]);

    for epoch in 1..max_epochs + 1 {
        let mut total_loss = 0.;
        let mut correct: usize = 0;
        tensors = optim.zero_grad(tensors);

        //let out = network.forward(x).view(n_shape);
    }

    todo!()
}

fn flatten(d: &[(f64, f64)]) -> Vec<f64> {
    d.iter()
        .fold(Vec::with_capacity(d.len() * 2), |mut acc, t| {
            acc.push(t.0);
            acc.push(t.1);
            acc
        })
}

#[derive(Debug, PartialEq)]
pub enum Result {
    Correct,
    Incorrect,
}

impl Result {
    fn to(&self) -> usize {
        match self {
            Result::Correct => 1,
            Result::Incorrect => 0,
        }
    }
}
