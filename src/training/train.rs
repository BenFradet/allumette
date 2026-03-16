use std::time::Instant;

use crate::{
    backend::{backend::Backend, mode::Mode},
    math::element::Element,
    optim::gradient_descent::GradientDescent,
    shaping::shape::Shape,
    tensor::Tensor,
    training::debugger::Debugger,
};

use super::{dataset::Dataset, network::Network};

pub fn train<'a, B: Backend + 'a, D: Debugger<'a, B>>(
    data: Dataset<B::Element>,
    learning_rate: B::Element,
    iterations: usize,
    hidden_layer_size: usize,
    debugger: &mut D,
) {
    let mut network = Network::new(hidden_layer_size);
    let sgd = GradientDescent::new(learning_rate);

    let features = data.features();
    let labels = data.labels();
    let n = data.n();
    let ones = data.ones();
    let one = Tensor::from_scalar(B::Element::one());

    let one_shape = Shape::scalar(1);
    let n_shape = data.n_shape();

    let start_time = Instant::now();

    for iteration in 1..iterations + 1 {
        // c.f. https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        let out = network.forward(features.clone()).view(&n_shape);
        let prob = (out.clone() * labels.clone())
            + (out.clone() - ones.clone()) * (labels.clone() - ones.clone());

        let loss = -prob.ln();

        let loss_loss = (loss.clone() / n.clone()).sum(None).view(&one_shape);
        let gradients = loss_loss.backprop(one.clone());

        network.step(&sgd, &gradients);

        debugger.debug(&loss, &labels, &out, (iteration, iterations), start_time);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend::{CpuSeqBackend, GpuBackend},
        storage::data::Data,
    };

    use super::*;

    #[test]
    fn test_train() {
        let xc: Tensor<CpuSeqBackend> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yc: Tensor<CpuSeqBackend> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
        let pc = (xc.clone() * yc.clone())
            + (xc.clone() - Tensor::from_scalar(1.) * (yc - Tensor::from_scalar(1.)));
        let lc = -pc.ln();
        let oc = lc.sum(None);
        let mc = oc.backward();
        let xcg = mc.wrt(&xc).unwrap().data.collect();
        assert_eq!(
            vec![-6.666666666666667, -3.333333333333333, -2.361111111111111],
            xcg
        );

        let xg: Tensor<GpuBackend> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yg: Tensor<GpuBackend> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
        let pg = (xg.clone() * yg.clone())
            + (xg.clone() - Tensor::from_scalar(1.) * (yg - Tensor::from_scalar(1.)));
        let lg = -pg.ln();
        let og = lg.sum(None);
        let mg = og.backward();
        let xgg = mg.wrt(&xg).unwrap().data.collect();
        assert_eq!(vec![-6.6666665, -3.3333335, -2.3611112], xgg);
    }
}
