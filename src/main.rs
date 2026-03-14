use std::io::Error;

use allumette::{
    backend::backend::CpuSeqBackend,
    training::{dataset::Dataset, debugger::VizDebugger, train},
};

fn main() -> Result<(), Error> {
    let pts = 500;
    let dataset = Dataset::star(pts);
    let hidden_layer_size = 50;
    let learning_rate = 0.2;
    let iterations = 500;

    let mut debugger = VizDebugger::new(&dataset, iterations);

    let mut debugger_thread_clone = debugger.clone();

    std::thread::spawn(move || {
        train::train::<CpuSeqBackend, _>(
            dataset,
            learning_rate,
            iterations,
            hidden_layer_size,
            &mut debugger_thread_clone,
        );
    });

    debugger.run()
}
