use crate::{
    autodiff::history::History,
    backend::{backend::Backend, mode::Mode},
    math::element::Element,
    shaping::shape::Shape,
    storage::data::Data,
    tensor::Tensor,
};

pub struct Layer<'a, B: Backend> {
    pub name: &'a str,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Tensor<'a, B>,
    pub biases: Tensor<'a, B>,
}

// TODO: make a gpu-specific impl if cpu doesn't work
impl<'a, B: Backend> Layer<'a, B> {
    pub fn new(name: &'a str, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
            //weights: Self::weights(name, in_size, out_size),
            //biases: Self::biases(name, out_size),
            weights: Self::weights_gpu(name, in_size, out_size),
            biases: Self::biases_gpu(name, out_size),
        }
    }

    pub fn forward(&self, t: Tensor<'a, B>) -> Tensor<'a, B> {
        let input_shape = t.data.shape().data().to_vec();
        let batch = input_shape[0];
        let in_size = input_shape[1];

        t.view(&Shape::new(vec![batch, in_size]))
            .mm(self.weights.clone())
            + self.biases.clone()
        //(self
        //    .weights
        //    .clone()
        //    .view(&Shape::new(vec![1, in_size, self.out_size]))
        //    * t.view(&Shape::new(vec![batch, in_size, 1])))
        //.sum(Some(1))
        //.view(&Shape::new(vec![batch, self.out_size]))
        //    + self.biases.clone().view(&Shape::new(vec![self.out_size]))
    }

    pub fn update_weights(mut self, w: Tensor<'a, B>) -> Self {
        self.weights = w;
        self
    }

    pub fn update_biases(mut self, b: Tensor<'a, B>) -> Self {
        self.biases = b;
        self
    }

    pub fn weights(name: &str, in_size: usize, out_size: usize) -> Tensor<'a, B> {
        let id = Self::weights_key(name);
        Self::param(Shape::new(vec![in_size, out_size])).id(id)
    }

    pub fn weights_gpu(name: &str, in_size: usize, out_size: usize) -> Tensor<'a, B> {
        let id = Self::weights_key(name);
        let shape = Shape::new(vec![in_size, out_size]);
        let t = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::rand_with_seed(
            shape, 1234,
        ));
        (t - Tensor::from_scalar(B::Element::fromf(0.5)))
            .history(History::default())
            .id(id)
    }

    pub fn biases(name: &str, out_size: usize) -> Tensor<'a, B> {
        let id = Self::biases_key(name);
        Self::param(Shape::new(vec![out_size])).id(id)
    }

    pub fn biases_gpu(name: &str, out_size: usize) -> Tensor<'a, B> {
        let id = Self::biases_key(name);
        let shape = Shape::new(vec![out_size]);
        let t = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::zeros(shape));
        (t + Tensor::from_scalar(B::Element::fromf(0.1)))
            .history(History::default())
            .id(id)
    }

    fn param(shape: Shape) -> Tensor<'a, B> {
        let t = Tensor::from_data(<B::Storage<'a> as Data<B::Element>>::rand_with_seed(
            shape, 1234,
        ));
        ((t - Tensor::from_scalar(B::Element::fromf(0.5)))
            * Tensor::from_scalar(B::Element::fromf(2.)))
        .history(History::default())
    }

    pub fn wkey(&self) -> String {
        format!("{}_weights", self.name)
    }

    fn weights_key(name: &str) -> String {
        format!("{name}_weights")
    }

    fn biases_key(name: &str) -> String {
        format!("{name}_biases")
    }

    pub fn bkey(&self) -> String {
        format!("{}_biases", self.name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::backend::{CpuSeqBackend, GpuBackend},
        shaping::strides::Strides,
        storage::{cpu_data::CpuData, gpu_data::GpuData},
        wgpu::wgpu_context::get_wgpu_context,
    };

    use super::*;

    #[test]
    fn test_layer() {
        let shape = Shape::new(vec![2, 3]);
        let strides: Strides = (&shape).into();

        let tdg = GpuData::new(
            &[1., 2., 3., 4., 5., 6.],
            shape.clone(),
            strides.clone(),
            get_wgpu_context(),
        );
        let g: Tensor<GpuBackend> = Tensor::from_data(tdg);
        let glayer = Layer::new("layer", 2, 2);
        let gout = glayer.forward(g.clone());
        let gloss = gout.sum(None);
        let gres = gloss.backward();
        assert_eq!(
            vec![1., 1., 1., 1., 1., 1.],
            gres.get(&g.id)
                .unwrap()
                .grad
                .clone()
                .unwrap()
                .data
                .collect()
        );

        let tdc = CpuData::new(vec![1., 2., 3., 4., 5., 6.], shape.clone(), strides.clone());
        let c: Tensor<CpuSeqBackend> = Tensor::from_data(tdc);
        let clayer = Layer::new("layer", 2, 2);
        let cout = clayer.forward(c.clone());
        let closs = cout.sum(None);
        let cres = closs.backward();
        assert_eq!(
            vec![1., 1., 1., 1., 1., 1.],
            cres.get(&c.id)
                .unwrap()
                .grad
                .clone()
                .unwrap()
                .data
                .collect()
        );
    }
}
