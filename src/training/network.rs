use crate::{
    autodiff::trace::Trace,
    backend::{backend::Backend, mode::Mode},
    optim::optimizer::Optimizer,
    tensor::Tensor,
};

use super::layer::Layer;

pub struct Network<'a, B: Backend> {
    layer1: Layer<'a, B>,
    layer2: Layer<'a, B>,
    layer3: Layer<'a, B>,
}

impl<'a, B: Backend> Network<'a, B> {
    pub fn new(hidden_layer_size: usize) -> Self {
        let layer1 = Layer::new("layer1", 2, hidden_layer_size);
        let layer2 = Layer::new("layer2", hidden_layer_size, hidden_layer_size);
        let layer3 = Layer::new("layer3", hidden_layer_size, 1);
        Self {
            layer1,
            layer2,
            layer3,
        }
    }

    //pub fn update(&mut self, tensors: &HashMap<String, Tensor<'a, B>>) {
    //    let l1_wkey = self.layer1.wkey();
    //    let l1_bkey = self.layer1.bkey();
    //    self.layer1.weights = tensors.get(&l1_wkey).unwrap().clone();
    //    self.layer1.biases = tensors.get(&l1_bkey).unwrap().clone();
    //    let l2_wkey = self.layer2.wkey();
    //    let l2_bkey = self.layer2.bkey();
    //    self.layer2.weights = tensors.get(&l2_wkey).unwrap().clone();
    //    self.layer2.biases = tensors.get(&l2_bkey).unwrap().clone();
    //    let l3_wkey = self.layer3.wkey();
    //    let l3_bkey = self.layer3.bkey();
    //    self.layer3.weights = tensors.get(&l3_wkey).unwrap().clone();
    //    self.layer3.biases = tensors.get(&l3_bkey).unwrap().clone();
    //}

    // TODO: find a way to remove the graph walk
    pub fn update(&mut self, root: &Tensor<'a, B>) {
        let sorted: Vec<_> = root.topological_sort_dfs().collect();
        for layer in [&mut self.layer1, &mut self.layer2, &mut self.layer3] {
            for param in [&mut layer.weights, &mut layer.biases] {
                if let Some(leaf) = sorted.iter().find(|t| t.id == param.id) {
                    *param.grad.borrow_mut() = leaf.grad.borrow().clone();
                }
            }
        }
    }

    pub fn forward(&self, x: Tensor<'a, B>) -> Tensor<'a, B> {
        let l1 = self.layer1.forward(x).relu();
        let l2 = self.layer2.forward(l1).relu();
        self.layer3.forward(l2).sig()
    }
}

impl<'a, B: Backend> Optimizer<'a, B> for Network<'a, B> {
    fn zero(&mut self) {
        *self.layer1.weights.grad.borrow_mut() = None;
        *self.layer2.weights.grad.borrow_mut() = None;
        *self.layer3.weights.grad.borrow_mut() = None;
        *self.layer1.biases.grad.borrow_mut() = None;
        *self.layer2.biases.grad.borrow_mut() = None;
        *self.layer3.biases.grad.borrow_mut() = None;
    }

    fn step(&mut self, lr_tensor: Tensor<'a, B>) {
        let l1w = self
            .layer1
            .weights
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l1w {
            let update = lr_tensor.clone() * grad;
            self.layer1.weights = (self.layer1.weights.clone() - update)
                .trace(Trace::default())
                .id(self.layer1.weights.id.clone());
        }
        let l1b = self
            .layer1
            .biases
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l1b {
            let update = lr_tensor.clone() * grad;
            self.layer1.biases = (self.layer1.biases.clone() - update)
                .trace(Trace::default())
                .id(self.layer1.biases.id.clone());
        }
        let l2w = self
            .layer2
            .weights
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l2w {
            let update = lr_tensor.clone() * grad;
            self.layer2.weights = (self.layer2.weights.clone() - update)
                .trace(Trace::default())
                .id(self.layer2.weights.id.clone());
        }
        let l2b = self
            .layer2
            .biases
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l2b {
            let update = lr_tensor.clone() * grad;
            self.layer2.biases = (self.layer2.biases.clone() - update)
                .trace(Trace::default())
                .id(self.layer2.biases.id.clone());
        }
        let l3w = self
            .layer3
            .weights
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l3w {
            let update = lr_tensor.clone() * grad;
            self.layer3.weights = (self.layer3.weights.clone() - update)
                .trace(Trace::default())
                .id(self.layer3.weights.id.clone());
        }
        let l3b = self
            .layer3
            .biases
            .grad
            .borrow()
            .as_ref()
            .map(|g| *g.clone());
        if let Some(grad) = l3b {
            let update = lr_tensor.clone() * grad;
            self.layer3.biases = (self.layer3.biases.clone() - update)
                .trace(Trace::default())
                .id(self.layer3.biases.id.clone());
        }
    }
}
