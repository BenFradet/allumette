use allumette::training::{dataset::Dataset, train};

fn main() -> () {
    let pts = 10;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.5;
    let iterations = 200;
    train::train(dataset, learning_rate, iterations, hidden_layer_size);
}
