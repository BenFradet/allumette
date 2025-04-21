use allumette::{data::dataset::Dataset, scalar::train::train_scalar, train::train_tensor};

fn main() -> () {
    let pts = 10;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.5;
    let max_epochs = 1;
    println!("scalars:");
    train_scalar::train(
        dataset.clone(),
        learning_rate,
        max_epochs,
        hidden_layer_size,
    );
    println!("\n\ntensors:");
    train_tensor::train(dataset, learning_rate, max_epochs, hidden_layer_size);
}
