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

impl<'a, E: Element + UnsafeUsizeConvert, BT: BackendType, T: Backend<E, BT>> Layer<'a, E, BT, T> {
    pub fn new(name: &'a str, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
            weights: Self::weights(name, in_size, out_size),
            biases: Self::biases(name, out_size),
        }
    }

    pub fn forward(&self, t: Tensor<E, BT, T>) -> Tensor<E, BT, T> {
        let input_shape = t.data.shape().data().to_vec();
        let batch = input_shape[0];
        let in_size = input_shape[1];

        t.view(&Shape::new(vec![batch, in_size]))
            .mm(self.weights.clone())
            + self.biases.clone()
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

    pub fn biases(name: &str, out_size: usize) -> Tensor<E, BT, T> {
        let id = Self::biases_key(name);
        Self::param(Shape::new(vec![out_size])).id(id)
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
