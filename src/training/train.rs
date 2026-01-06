use rand::{seq::SliceRandom, thread_rng};

use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    data::tensor_data::TensorData,
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

    let x = Tensor::from_tuples(&data.x);
    let y_data = <T as TensorData<E>>::from_1d(
        &data
            .y
            .iter()
            .map(|u| E::fromf(*u as f64))
            .collect::<Vec<_>>(),
    );
    let y = Tensor::from_data(y_data);
    let n_shape = Shape::new(vec![data.n]);
    let one_shape = Shape::scalar(1);

    for iteration in 1..iterations + 1 {
        network.zero();

        let out = network.forward(x.clone()).view(&n_shape);
        let prob = (out.clone() * y.clone())
            + (out.clone() - Tensor::from_scalar(E::one()))
                * (y.clone() - Tensor::from_scalar(E::one()));

        let loss = -prob.ln();

        let res = (loss.clone() / Tensor::from_scalar(E::fromf(data.n as f64)))
            .sum(None)
            .view(&one_shape)
            .backward();
        network.update(&res);

        let total_loss = loss
            .clone()
            .sum(None)
            .view(&one_shape)
            .item()
            .unwrap_or(E::zero());

        network.step(lr_tensor.clone());

        if iteration.is_multiple_of(10) || iteration == iterations {
            let y2 = y.clone();
            let correct = out
                .clone()
                .gt(Tensor::from_scalar(E::fromf(0.5)))
                .eq(y2)
                .sum(None)
                .item()
                .unwrap();
            println!("Iteration {iteration}, loss: {total_loss}, correct: {correct}\n");
        }
    }
}

pub fn train_shuffle<E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>>(
    data: Dataset<E>,
    learning_rate: E,
    iterations: usize,
    hidden_layer_size: usize,
) {
    let mut network: Network<'_, E, BT, T> = Network::new(hidden_layer_size);
    let lr_tensor = Tensor::from_scalar(learning_rate);
    let mut rng = thread_rng();
    let yd: Vec<_> = data.y.iter().map(|u| E::fromf(*u as f64)).collect();
    let n_shape = Shape::new(vec![data.n]);
    let one_shape = Shape::scalar(1);

    let xt = Tensor::from_tuples(&data.x);
    let yt = Tensor::from_1d(&yd);

    let mut c: Vec<_> = data.x.iter().zip(yd).collect();
    for iteration in 1..=iterations {
        c.shuffle(&mut rng);
        let (x_shuf, y_shuf): (Vec<_>, Vec<_>) = c.clone().into_iter().unzip();

        network.zero();
        let y = Tensor::from_1d(&y_shuf);
        let x = Tensor::from_tuples(&x_shuf);

        let out = network.forward(x.clone()).view(&n_shape);
        let prob = (out.clone() * y.clone())
            + (out.clone() - Tensor::from_scalar(E::one()))
                * (y.clone() - Tensor::from_scalar(E::one()));

        let loss = -prob.clone().ln();

        let res = (loss.clone() / Tensor::from_scalar(E::fromf(data.n as f64)))
            .sum(None)
            .view(&one_shape)
            .backward();
        network.update(&res);

        let total_loss = loss
            .clone()
            .sum(None)
            .view(&one_shape)
            .item()
            .unwrap_or(E::zero());

        network.step(lr_tensor.clone());

        if iteration.is_multiple_of(10) || iteration == iterations {
            let out = network.forward(xt.clone()).view(&n_shape);
            let y2 = yt.clone();
            let correct = out
                .clone()
                .gt(Tensor::from_scalar(E::fromf(0.5)))
                .eq(y2)
                .sum(None)
                .item()
                .unwrap();
            println!("Iteration {iteration}, loss: {total_loss}, correct: {correct}\n");
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
        println!("xcg {xcg:?}");

        let xg: Tensor<_, Gpu, GpuTensorData> = Tensor::from_1d(&[0.2, 0.5, 0.8]);
        let yg: Tensor<_, Gpu, GpuTensorData> = Tensor::from_2d(&[&[0.], &[1.], &[0.]]).unwrap();
        let xg_id = xg.id.clone();
        let pg = (xg.clone() * yg.clone())
            + (xg - Tensor::from_scalar(1.) * (yg - Tensor::from_scalar(1.)));
        let lg = -pg.ln();
        let og = lg.sum(None);
        let mg = og.backward();
        let xgg = mg.get(&xg_id).unwrap().grad.clone().unwrap().data.collect();
        println!("xgg {xgg:?}");
    }
}
