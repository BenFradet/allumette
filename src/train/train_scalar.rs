use crate::{data::graph::Graph, module::module::Module, optim::{optimizer::Optimizer, sgd::SGD}, scalar::scalar::Scalar};

use super::{linear::Linear, network::Network};

fn train(
    data: Graph,
    learning_rate: f64,
    max_epochs: usize,
    hidden_layer_size: usize,
) -> () {
    let mut module = Module::default();
    let model = Network::new(&mut module, hidden_layer_size);
    let mut optim = SGD::new(&mut module, learning_rate);
    let mut losses = vec![];
    for epoch in 1..max_epochs + 1 {
        let mut total_loss = 0.;
        let mut correct: usize = 0;
        optim.zero();

        for (i, (x1, x2)) in data.x.iter().enumerate() {
            let label = data.y[i];
            let x1_scalar = Scalar::new(*x1);
            let x2_scalar = Scalar::new(*x2);
            let out = model.forward(x1_scalar, x2_scalar);

            let (prob, result) = prob(out, label);
            correct += result.to();

            let loss = -prob.ln();
            let loss_v = loss.v;
            let res = (loss / &Scalar::new(data.n as f64)).backprop(1.);
            // TODO: update params
            total_loss += loss_v;
        }

        losses.push(total_loss);
        optim.step();

        if epoch % 10 == 0 || epoch == max_epochs {
            print!("Epoch {epoch}, loss: {total_loss}, correct: {correct}");
        }
    }
}

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

fn prob(out: Option<Scalar>, label: usize) -> (Scalar, Result) {
    let output = out.unwrap_or(Scalar::new(0.));
    let v = output.v;
    match label {
        1 => (output, if v > 0.5 { Result::Correct } else { Result::Incorrect }),
        _ => {
            let prob = -output + Scalar::new(1.);
            (prob, if v < 0.5 { Result:: Correct} else { Result::Incorrect })
        },
    }
}