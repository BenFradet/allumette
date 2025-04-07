use crate::data::dataset::Dataset;

use super::network::Network;

pub fn train(data: Dataset, learning_rate: f64, max_epochs: usize, hidden_layer_size: usize) -> () {
    let network = Network::new(hidden_layer_size);
    todo!()
}

#[derive(Debug, PartialEq)]
pub enum Result {
    Correct,
    Incorrect,
}

impl Result {
    fn to(&self) -> usize {
        match self {
            Result::Correct => 1,
            Result::Incorrect => 0,
        }
    }
}
