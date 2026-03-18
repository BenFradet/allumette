use std::io::Error;

use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend, GpuBackend},
    training::{
        dataset::Dataset,
        debugger::{TerseDebugger, VizDebugger},
        train,
    },
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand)]
enum CliCommand {
    Benchmark {
        #[arg(value_enum, short, long)]
        backend: CliBackend,
        #[arg(short, long, default_value_t = 1000)]
        points: usize,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum CliBackend {
    Seq,
    Par,
    Gpu,
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    let hidden_layer_size = 50;
    let learning_rate = 0.2;
    let iterations = 500;

    match cli.command {
        Some(CliCommand::Benchmark { backend, points }) => {
            benchmark(
                backend,
                points,
                hidden_layer_size,
                learning_rate,
                iterations,
            );
            Ok(())
        }
        _ => viz(hidden_layer_size, learning_rate, iterations),
    }
}

fn viz(hidden_layer_size: usize, learning_rate: f64, iterations: usize) -> Result<(), Error> {
    let pts = 1000;
    let dataset = Dataset::circle(pts);

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

fn benchmark(
    backend: CliBackend,
    points: usize,
    hidden_layer_size: usize,
    learning_rate: f64,
    iterations: usize,
) {
    for run in 0..5 {
        if run == 0 {
            println!("run 0/5 backend={backend:?} points={points} - ignore as warm up");
        } else {
            println!("run {run}/5 backend={backend:?} points={points}");
        }
        match backend {
            CliBackend::Seq => {
                let dataset = Dataset::circle(points);
                train::train::<CpuSeqBackend, _>(
                    dataset,
                    learning_rate,
                    iterations,
                    hidden_layer_size,
                    &mut TerseDebugger {},
                );
            }
            CliBackend::Par => {
                let dataset = Dataset::circle(points);
                train::train::<CpuParBackend, _>(
                    dataset,
                    learning_rate,
                    iterations,
                    hidden_layer_size,
                    &mut TerseDebugger {},
                );
            }
            CliBackend::Gpu => {
                let dataset = Dataset::circle(points);
                train::train::<GpuBackend, _>(
                    dataset,
                    learning_rate as f32,
                    iterations,
                    hidden_layer_size,
                    &mut TerseDebugger {},
                );
            }
        }
    }
}
