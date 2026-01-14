use std::time::Instant;

use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    optim::optimizer::Optimizer,
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
) {
    let mut network = Network::new(hidden_layer_size);
    let lr_tensor = Tensor::from_scalar(learning_rate);

    let x = data.x();
    let y = data.y();
    let n = data.n();
    let ones = data.ones();
    let one = Tensor::from_scalar(B::Element::one());

    let one_shape = Shape::scalar(1);
    let n_shape = data.n_shape();

    let start_time = Instant::now();

    for iteration in 1..iterations + 1 {
        network.zero();

        let out = network.forward(x.clone()).view(&n_shape);
        let prob =
            (out.clone() * y.clone()) + (out.clone() - ones.clone()) * (y.clone() - ones.clone());

        let loss = -prob.ln();

        let res = (loss.clone() / n.clone())
            .sum(None)
            .view(&one_shape)
            .backprop(one.clone());

        network.update(&res);
        network.step(lr_tensor.clone());

        D::debug(&loss, &y, &out, (iteration, iterations), start_time);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend::{CpuSeqBackend, GpuBackend}, data::tensor_data::TensorData,
    };

    use super::*;

    #[test]
    fn test_train() {
        let xc: Tensor<CpuSeqBackend> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yc: Tensor<CpuSeqBackend> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
        let xc_id = xc.id.clone();
        let pc = (xc.clone() * yc.clone())
            + (xc - Tensor::from_scalar(1.) * (yc - Tensor::from_scalar(1.)));
        let lc = -pc.ln();
        let oc = lc.sum(None);
        let mc = oc.backward();
        let xcg = mc.get(&xc_id).unwrap().grad.clone().unwrap().data.collect();
        assert_eq!(
            vec![-6.666666666666667, -3.333333333333333, -2.361111111111111],
            xcg
        );

        let xg: Tensor<GpuBackend> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yg: Tensor<GpuBackend> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
        let xg_id = xg.id.clone();
        let pg = (xg.clone() * yg.clone())
            + (xg - Tensor::from_scalar(1.) * (yg - Tensor::from_scalar(1.)));
        let lg = -pg.ln();
        let og = lg.sum(None);
        let mg = og.backward();
        let xgg = mg.get(&xg_id).unwrap().grad.clone().unwrap().data.collect();
        assert_eq!(vec![-6.6666665, -3.3333335, -2.3611112], xgg);
    }
}
