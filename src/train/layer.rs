use crate::{
    backend::{backend::Backend, backend_type::BackendType},
    shaping::shaped::Shaped,
    tensor::{
        shaping::shape::Shape, tensor::Tensor, tensor_data::TensorData, tensor_history::History,
    },
};

pub struct Layer<'a, BT: BackendType, T: Backend<BT>> {
    pub name: &'a str,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: Tensor<BT, T>,
    pub biases: Tensor<BT, T>,
}

impl<
        'a,
        BT: BackendType + Clone + std::fmt::Debug,
        T: Backend<BT> + Shaped + Clone + std::fmt::Debug,
    > Layer<'a, BT, T>
{
    pub fn new(name: &'a str, in_size: usize, out_size: usize) -> Self {
        Self {
            name,
            in_size,
            out_size,
            weights: Self::weights(name, in_size, out_size),
            biases: Self::biases(name, out_size),
        }
    }

    pub fn forward(&self, t: Tensor<BT, T>) -> Tensor<BT, T> {
        let mut input_shape = t.data.shape().data().to_vec();
        let input_first_dim = input_shape[0];

        let mut weights_shape = self.weights.data.shape().data().to_vec();

        // input size must match weight size
        assert!(
            input_shape.last() == weights_shape.first(),
            "input size does not match weight size"
        );

        // reshape weights to prepare for matrix multiplication by adding a batch dimension
        weights_shape.insert(0, 1);
        let reshaped_weights = self.weights.clone().view(&Shape::new(weights_shape));

        // reshape input tensor by adding an output dimension
        input_shape.push(1);
        let reshaped_input = t.view(&Shape::new(input_shape));

        // perform element-wise multiplication and then sum across the input dimension
        // to perform a dot product equivalent for each sample in the batch
        let batch_product = reshaped_weights * reshaped_input;
        let summed_product = batch_product.sum(Some(1)).contiguous();

        // reshape the summed product to match the output dimension
        // number of samples by output size
        let output_shape = vec![input_first_dim, self.out_size];
        let reshaped_output = summed_product.view(&Shape::new(output_shape));

        // add bias to each output in the batch
        // bias is reshaped to match the batch output shape (1, out_size) for broadcasting
        reshaped_output
            + self
                .biases
                .clone()
                .view(&Shape::new(vec![1, self.out_size]))
    }

    pub fn update_weights(mut self, w: Tensor<BT, T>) -> Self {
        self.weights = w;
        self
    }

    pub fn update_biases(mut self, b: Tensor<BT, T>) -> Self {
        self.biases = b;
        self
    }

    //pub fn frozen_weights(name: &str, in_size: usize, out_size: usize) -> Tensor<BT, T> {
    //    let d = if name == "layer1" {
    //        vec![-0.98, -0.20, 0.46, 0.94, 0.22, 0.29]
    //    } else if name == "layer2" {
    //        vec![-0.16, -0.47, -0.58, 0.40, -0.79, -0.48, -0.86, -0.85, -0.90]
    //    } else {
    //        vec![-0.21, 0.91, -0.79]
    //    };
    //    let shape = Shape::new(vec![in_size, out_size]);
    //    let strides = (&shape).into();
    //    let td = TensorData::new(d, shape, strides);
    //    let id = Self::weights_key(name);
    //    Tensor::from_data(td)
    //        .history(History::default())
    //        .id(id)
    //}

    //pub fn frozen_biases(name: &str, out_size: usize) -> Tensor<BT, T> {
    //    let d = if name == "layer1" {
    //        vec![-0.17, -0.38, 0.84]
    //    } else if name == "layer2" {
    //        vec![0.5, -0.48, 0.83]
    //    } else {
    //        vec![-0.04]
    //    };
    //    let shape = Shape::new(vec![out_size]);
    //    let strides = (&shape).into();
    //    let td = TensorData::new(d, shape, strides);
    //    let id = Self::biases_key(name);
    //    Tensor::from_data(td)
    //        .history(History::default())
    //        .id(id)
    //}

    pub fn weights(name: &str, in_size: usize, out_size: usize) -> Tensor<BT, T> {
        let id = Self::weights_key(name);
        Self::param(Shape::new(vec![in_size, out_size])).id(id)
    }

    pub fn biases(name: &str, out_size: usize) -> Tensor<BT, T> {
        let id = Self::biases_key(name);
        Self::param(Shape::new(vec![out_size])).id(id)
    }

    fn param(shape: Shape) -> Tensor<BT, T> {
        let t = Tensor::from_data(<T as Shaped>::rand(shape));
        ((t - Tensor::scalar(0.5)) * Tensor::scalar(2.)).history(History::default())
    }

    pub fn wkey(&self) -> String {
        format!("{}_weights", self.name)
    }

    fn weights_key(name: &str) -> String {
        format!("{}_weights", name)
    }

    fn biases_key(name: &str) -> String {
        format!("{}_biases", name)
    }

    pub fn bkey(&self) -> String {
        format!("{}_biases", self.name)
    }
}
