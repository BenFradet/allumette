use crate::{
    autodiff::trace::Trace,
    backend::{backend::Backend, mode::Mode},
    math::element::Element,
    shaping::shape::Shape,
    storage::data::Data,
    tensor::Tensor,
};

pub struct Layer<'a, B: Backend> {
    pub weights: Tensor<'a, B>,
    pub biases: Tensor<'a, B>,
}

impl<'a, B: Backend> Layer<'a, B> {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        Self {
            weights: Self::weights(in_size, out_size),
            biases: Self::biases(out_size),
        }
    }

    pub fn forward(&self, t: Tensor<'a, B>) -> Tensor<'a, B> {
        let input_shape = t.data.shape().data().to_vec();
        let batch = input_shape[0];
        let in_size = input_shape[1];

        t.view(&Shape::new(vec![batch, in_size]))
            .mm(self.weights.clone())
            + self.biases.clone()
    }

    pub fn update_weights(mut self, w: Tensor<'a, B>) -> Self {
        self.weights = w;
        self
    }

    pub fn update_biases(mut self, b: Tensor<'a, B>) -> Self {
        self.biases = b;
        self
    }

    pub fn weights(in_size: usize, out_size: usize) -> Tensor<'a, B> {
        let shape = Shape::new(vec![in_size, out_size]);
        let t = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::rand(shape));
        (t - Tensor::from_scalar(B::Element::fromf(0.5))).trace(Trace::default())
    }

    pub fn biases(out_size: usize) -> Tensor<'a, B> {
        let shape = Shape::new(vec![out_size]);
        let t = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::zeros(shape));
        (t + Tensor::from_scalar(B::Element::fromf(0.1))).trace(Trace::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend::CpuSeqBackend, shaping::strides::Strides, storage::cpu_data::CpuData,
    };
    #[cfg(feature = "gpu")]
    use crate::{
        backend::backend::GpuBackend, storage::gpu_data::GpuData,
        wgpu::wgpu_context::get_wgpu_context,
    };

    use super::*;

    #[test]
    fn test_layer_cpu() {
        let shape = Shape::new(vec![2, 3]);
        let strides: Strides = (&shape).into();

        let tdc = CpuData::new(vec![1., 2., 3., 4., 5., 6.], shape.clone(), strides.clone());
        let c: Tensor<CpuSeqBackend> = Tensor::from_data(tdc);
        let clayer = Layer::new(2, 2);
        let cout = clayer.forward(c.clone());
        let closs = cout.sum(None);
        let cres = closs.backward();
        assert_eq!(
            vec![1., 1., 1., 1., 1., 1.],
            cres.wrt(&c).unwrap().data.collect()
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_layer_gpu() {
        let shape = Shape::new(vec![2, 3]);
        let strides: Strides = (&shape).into();

        let tdg = GpuData::new(
            &[1., 2., 3., 4., 5., 6.],
            shape.clone(),
            strides.clone(),
            get_wgpu_context(),
        );
        let g: Tensor<GpuBackend> = Tensor::from_data(tdg);
        let glayer = Layer::new(2, 2);
        let gout = glayer.forward(g.clone());
        let gloss = gout.sum(None);
        let gres = gloss.backward();
        assert_eq!(
            vec![1., 1., 1., 1., 1., 1.],
            gres.wrt(&g).unwrap().data.collect()
        );
    }
}
