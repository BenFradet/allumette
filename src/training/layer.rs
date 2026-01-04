use crate::{
    autodiff::history::History,
    backend::{backend::Backend, backend_type::BackendType},
    data::tensor_data::TensorData,
    math::element::Element,
    shaping::shape::Shape,
    tensor::Tensor,
    util::unsafe_usize_convert::UnsafeUsizeConvert,
};

pub struct Layer<'a, E: Element, BT: BackendType, T: Backend<E, BT>> {
    pub name: &'a str,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Tensor<E, BT, T>,
    pub biases: Tensor<E, BT, T>,
}

// TODO: make a gpu-specific impl if cpu doesn't work
impl<'a, E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Layer<'a, E, BT, T> {
    pub fn new(name: &'a str, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
            weights: Self::weights_gpu(name, in_size, out_size),
            biases: Self::biases_gpu(name, out_size),
        }
    }

    pub fn forward(&self, t: Tensor<E, BT, T>) -> Tensor<E, BT, T> {
        let input_shape = t.data.shape().data().to_vec();
        let batch = input_shape[0];
        let in_size = input_shape[1];

        //t.view(&Shape::new(vec![batch, in_size]))
        //    .mm(self.weights.clone())
        //    + self.biases.clone()
        (self
            .weights
            .clone()
            .view(&Shape::new(vec![1, in_size, self.out_size]))
            * t.view(&Shape::new(vec![batch, in_size, 1])))
        .sum(Some(1))
        .view(&Shape::new(vec![batch, self.out_size]))
            + self.biases.clone().view(&Shape::new(vec![self.out_size]))
    }

    pub fn update_weights(mut self, w: Tensor<E, BT, T>) -> Self {
        self.weights = w;
        self
    }

    pub fn update_biases(mut self, b: Tensor<E, BT, T>) -> Self {
        self.biases = b;
        self
    }

    pub fn weights(name: &str, in_size: usize, out_size: usize) -> Tensor<E, BT, T> {
        let id = Self::weights_key(name);
        Self::param(Shape::new(vec![in_size, out_size])).id(id)
    }

    pub fn weights_gpu(name: &str, in_size: usize, out_size: usize) -> Tensor<E, BT, T> {
        let id = Self::weights_key(name);
        let shape = Shape::new(vec![in_size, out_size]);
        let t = Tensor::from_data(<T as TensorData<E>>::rand_with_seed(shape, 1234));
        (t - Tensor::from_scalar(E::fromf(0.5)))
            .history(History::default())
            .id(id)
    }

    pub fn biases(name: &str, out_size: usize) -> Tensor<E, BT, T> {
        let id = Self::biases_key(name);
        Self::param(Shape::new(vec![out_size])).id(id)
    }

    pub fn biases_gpu(name: &str, out_size: usize) -> Tensor<E, BT, T> {
        let id = Self::biases_key(name);
        let shape = Shape::new(vec![out_size]);
        let t = Tensor::from_data(<T as TensorData<E>>::zeros(shape));
        (t + Tensor::from_scalar(E::fromf(0.1)))
            .history(History::default())
            .id(id)
    }

    fn param(shape: Shape) -> Tensor<E, BT, T> {
        let t = Tensor::from_data(<T as TensorData<E>>::rand(shape));
        ((t - Tensor::from_scalar(E::fromf(0.5))) * Tensor::from_scalar(E::fromf(2.)))
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
        backend::backend_type::Seq,
        data::{cpu_tensor_data::CpuTensorData, gpu_tensor_data::GpuTensorData},
        shaping::strides::Strides,
        wgpu::wgpu_context::get_wgpu_context,
    };

    use super::*;

    #[test]
    fn layer_test() {
        let shape = Shape::new(vec![2, 3]);
        let strides: Strides = (&shape).into();

        let tdg = GpuTensorData::new(
            &[1., 2., 3., 4., 5., 6.],
            shape.clone(),
            strides.clone(),
            get_wgpu_context(),
        );
        let g = Tensor::from_data(tdg);
        let glayer = Layer::new("layer", 2, 2);
        let gout = glayer.forward(g.clone());
        let gloss = gout.sum(None);
        let gres = gloss.backward();
        println!(
            "{:?}",
            gres.get(&g.id)
                .unwrap()
                .grad
                .clone()
                .unwrap()
                .data
                .collect()
        );

        let tdc = CpuTensorData::new(vec![1., 2., 3., 4., 5., 6.], shape.clone(), strides.clone());
        let c: Tensor<f64, Seq, _> = Tensor::from_data(tdc);
        let clayer: Layer<_, Seq, _> = Layer::new("layer", 2, 2);
        let cout = clayer.forward(c.clone());
        let closs = cout.sum(None);
        let cres = closs.backward();
        println!(
            "{:?}",
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
