use crate::{
    data::dataset::Dataset,
    optim::{optimizer::Optimizer, sgd::SGD},
    scalar::scalar::Scalar,
};

use super::network::Network;

// TODO: make it work with module
pub fn train(data: Dataset, learning_rate: f64, max_epochs: usize, hidden_layer_size: usize) -> () {
    let network = Network::new(hidden_layer_size);
    let mut scalars = network.init();
    let optim = SGD::new(learning_rate);
    let mut losses = vec![];

    for epoch in 1..max_epochs + 1 {
        let mut total_loss = 0.;
        let mut correct: usize = 0;
        scalars = optim.zero(scalars);

        for (i, (x1, x2)) in data.x.iter().enumerate() {
            let label = data.y[i];
            let x1_scalar = Scalar::new(*x1);
            let x2_scalar = Scalar::new(*x2);
            let out = network.forward(x1_scalar, x2_scalar, &scalars);

            let (prob, result) = prob(out, label);
            correct += result.to();

            let loss = -prob.ln();
            let loss_v = loss.v;
            let res = (loss / &Scalar::new(data.n as f64)).backprop(1.);
            scalars.extend(res);
            total_loss += loss_v;
        }

        losses.push(total_loss);
        scalars = optim.step(scalars);

        if epoch % 10 == 0 || epoch == max_epochs {
            println!("Epoch {epoch}, loss: {total_loss}, correct: {correct}");
        }
    }
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

fn prob(out: Option<Scalar>, label: usize) -> (Scalar, Result) {
    let output = out.unwrap_or(Scalar::new(0.));
    let v = output.v;
    match label {
        1 => (
            output,
            if v > 0.5 {
                Result::Correct
            } else {
                Result::Incorrect
            },
        ),
        _ => {
            let prob = -output + Scalar::new(1.);
            (
                prob,
                if v < 0.5 {
                    Result::Correct
                } else {
                    Result::Incorrect
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bimap<A, B, A1, B1, FA, FB>(t: (A, B), fa: FA, fb: FB) -> (A1, B1)
    where
        FA: Fn(A) -> A1,
        FB: Fn(B) -> B1,
    {
        let a1 = fa(t.0);
        let b1 = fb(t.1);
        (a1, b1)
    }

    #[test]
    fn prob_tests() -> () {
        let s1 = Scalar::new(1.);
        let s75 = Scalar::new(0.75);
        let s50 = Scalar::new(0.5);
        let s25 = Scalar::new(0.25);
        let s0 = Scalar::new(0.);
        assert_eq!((1., Result::Correct), bimap(prob(Some(s1.clone()), 1), |s| s.v, |r| r));
        assert_eq!((0.75, Result::Correct), bimap(prob(Some(s75.clone()), 1), |s| s.v, |r| r));
        assert_eq!((0.5, Result::Incorrect), bimap(prob(Some(s50.clone()), 1), |s| s.v, |r| r));
        assert_eq!((0.25, Result::Incorrect), bimap(prob(Some(s25.clone()), 1), |s| s.v, |r| r));
        assert_eq!((0., Result::Incorrect), bimap(prob(Some(s0.clone()), 1), |s| s.v, |r| r));
        assert_eq!((0., Result::Incorrect), bimap(prob(Some(s1), 0), |s| s.v, |r| r));
        assert_eq!((0.25, Result::Incorrect), bimap(prob(Some(s75), 0), |s| s.v, |r| r));
        assert_eq!((0.5, Result::Incorrect), bimap(prob(Some(s50), 0), |s| s.v, |r| r));
        assert_eq!((0.75, Result::Correct), bimap(prob(Some(s25), 0), |s| s.v, |r| r));
        assert_eq!((1., Result::Correct), bimap(prob(Some(s0), 0), |s| s.v, |r| r));
    }
}
