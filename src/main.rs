use std::io::Error;

use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend, GpuBackend},
    training::{
        dataset::Dataset,
        debugger::{ChattyDebugger, TerseDebugger, VizDebugger},
        train,
    },
};

fn main() -> Result<(), Error> {
    let pts = 1000;
    let dataset = Dataset::simple(pts);
    let hidden_layer_size = 3;
    let learning_rate = 0.1;
    let iterations = 200;

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

    let terminal = ratatui::init();
    let result = debugger.run(terminal);
    ratatui::restore();

    result
}
