use std::time::Instant;

use crate::{
    backend::{backend::Backend, mode::Mode},
    math::element::Element,
    optim::gradient_descent::GradientDescent,
    shaping::shape::Shape,
    tensor::Tensor,
    util::{debugger::Debugger, profiler::Profiler},
};

use super::{dataset::Dataset, network::Network};

pub fn train<'a, B: Backend + 'a, D: Debugger<'a, B>>(
    data: Dataset<B::Element>,
    learning_rate: B::Element,
    iterations: usize,
    hidden_layer_size: usize,
    debugger: &mut D,
    profiling_path: Option<&String>,
) {
    let mut network = Network::new(hidden_layer_size);
    let gd = GradientDescent::new(learning_rate);

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

        network.step(&gd, &gradients);

        debugger.debug(&loss, &labels, &out, (iteration, iterations), start_time);
    }

    if let Some(path) = profiling_path {
        <B::Profiler>::flush(path);
        println!("profiling information written out to {path}");
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu")]
    use crate::{
        autodiff::trace::Trace,
        backend::backend::{CpuSeqBackend, GpuBackend},
        shaping::strides::Strides,
        storage::{cpu_data::CpuData, data::Data, gpu_data::GpuData},
        wgpu::wgpu_context::get_wgpu_context,
    };
    #[cfg(feature = "gpu")]
    use serial_test::serial;

    #[cfg(feature = "gpu")]
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    #[serial(gpu)]
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

    #[cfg(feature = "gpu")]
    #[test]
    #[serial(gpu)]
    fn gpu_vs_cpu() {
        let n = 100;
        let data64: Vec<_> = (0..n * 2).map(|i| (i as f64) / (n * 2) as f64).collect();
        let data32: Vec<_> = data64.iter().map(|&i| i as f32).collect();

        let shape = Shape::new(vec![n, 2]);
        let strides: Strides = (&shape).into();

        let input_c: Tensor<CpuSeqBackend> =
            Tensor::from_data(CpuData::new(data64, shape.clone(), strides.clone()));
        let input_g: Tensor<GpuBackend> = Tensor::from_data(GpuData::new(
            &data32,
            shape.clone(),
            strides.clone(),
            get_wgpu_context(),
        ));

        let w_c: Tensor<CpuSeqBackend> = Tensor::from_2d(&[&[0.3, -0.2, 0.5], &[-0.3, 0.4, -0.1]])
            .unwrap()
            .trace(Trace::default());
        let w_g: Tensor<GpuBackend> = Tensor::from_2d(&[&[0.3, -0.2, 0.5], &[-0.3, 0.4, -0.1]])
            .unwrap()
            .trace(Trace::default());

        let b_c: Tensor<CpuSeqBackend> = Tensor::from_1d(&[0.1, 0.1, 0.1]).trace(Trace::default());
        let b_g: Tensor<GpuBackend> = Tensor::from_1d(&[0.1, 0.1, 0.1]).trace(Trace::default());

        let mm_c = input_c.clone().mm(w_c.clone());
        let mm_g = input_g.clone().mm(w_g.clone());
        compare("mm", &mm_c.data.collect(), &mm_g.data.collect(), 0.001);

        let add_c = mm_c + b_c.clone();
        let add_g = mm_g + b_g.clone();
        compare("add", &add_c.data.collect(), &add_g.data.collect(), 0.001);

        let relu_c = add_c.relu();
        let relu_g = add_g.relu();
        compare(
            "relu",
            &relu_c.data.collect(),
            &relu_g.data.collect(),
            0.001,
        );

        let sum_c = relu_c.sum(None);
        let sum_g = relu_g.sum(None);
        compare("sum", &sum_c.data.collect(), &sum_g.data.collect(), 0.01);

        let grads_c = sum_c.backward();
        let grads_g = sum_g.backward();

        compare(
            "w_grad",
            &grads_c.wrt(&w_c).unwrap().data.collect(),
            &grads_g.wrt(&w_g).unwrap().data.collect(),
            0.01,
        );
        compare(
            "b_grad",
            &grads_c.wrt(&b_c).unwrap().data.collect(),
            &grads_g.wrt(&b_g).unwrap().data.collect(),
            0.01,
        );
        compare(
            "input_grad",
            &grads_c.wrt(&input_c).unwrap().data.collect(),
            &grads_g.wrt(&input_g).unwrap().data.collect(),
            0.01,
        );
    }

    #[cfg(feature = "gpu")]
    fn compare(label: &str, cpu: &[f64], gpu: &[f32], tol: f64) {
        assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
        for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            let diff = (*c - *g as f64).abs();
            assert!(diff < tol, "{label}[{i}]: cpu={c}, gpu={g}, diff={diff}");
        }
    }
}
