use std::time::Instant;

use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    math::element::Element,
    optim::optimizer::Optimizer,
    shaping::shape::Shape,
    tensor::Tensor,
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

use super::{dataset::Dataset, network::Network};

pub fn train<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
    data: Dataset<E>,
    learning_rate: E,
    iterations: usize,
    hidden_layer_size: usize,
) {
    let mut network: Network<'_, E, BT, T> = Network::new(hidden_layer_size);
    let lr_tensor = Tensor::from_scalar(learning_rate);

    let x = data.x();
    let y = data.y();
    let n = data.n();
    let ones = data.ones();
    let one = Tensor::from_scalar(E::one());

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

        if iteration == iterations {
            let elapsed_time = start_time.elapsed();
            let total_loss = loss
                .clone()
                .sum(None)
                .view(&one_shape)
                .item()
                .unwrap_or(E::zero());

            let correct = out
                .clone()
                .gt(Tensor::from_scalar(E::fromf(0.5)))
                .eq(y.clone())
                .sum(None)
                .item()
                .unwrap();

            println!("elapsed time: {elapsed_time:?}, loss: {total_loss}, correct: {correct}");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend_type::{Gpu, Seq},
        data::{cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData},
    };

    use super::*;

    #[test]
    fn test_train() {
        let xc: Tensor<_, Seq, CpuTensorData> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yc: Tensor<_, Seq, CpuTensorData> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
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

        let xg: Tensor<_, Gpu, GpuTensorData> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yg: Tensor<_, Gpu, GpuTensorData> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
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
