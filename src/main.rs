use std::io::Error;

use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend, GpuBackend},
    training::{dataset::Dataset, train},
    util::{
        debugger::{TerseDebugger, VizDebugger},
        profiler::{CsvProfiler, NoopProfiler, Profiler},
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
        #[arg(short, long, default_value_t = 4)]
        power_ten_points: u32,
    },
    Profile {
        #[arg(value_enum, short, long)]
        backend: CliBackend,
        #[arg(short, long, default_value_t = 4)]
        power_ten_points: u32,
        #[arg(short, long)]
        output_path: String,
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
        Some(CliCommand::Benchmark {
            backend,
            power_ten_points,
        }) => {
            benchmark(
                backend,
                power_ten_points,
                hidden_layer_size,
                learning_rate,
                iterations,
            );
            Ok(())
        }
        Some(CliCommand::Profile {
            backend,
            power_ten_points,
            output_path: profile_path,
        }) => {
            profile(
                backend,
                power_ten_points,
                hidden_layer_size,
                learning_rate,
                iterations,
                profile_path,
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
            None,
        );
    });

    debugger.run()
}

fn profile(
    backend: CliBackend,
    power_ten_points: u32,
    hidden_layer_size: usize,
    learning_rate: f64,
    iterations: usize,
    profile_path: String,
) {
    let points = 10_usize.pow(power_ten_points);
    run_training::<CsvProfiler>(
        &backend,
        points,
        hidden_layer_size,
        learning_rate,
        iterations,
        Some(&profile_path),
    )
}

fn benchmark(
    backend: CliBackend,
    power_ten_points: u32,
    hidden_layer_size: usize,
    learning_rate: f64,
    iterations: usize,
) {
    let points = 10_usize.pow(power_ten_points);
    for run in 1..=5 {
        if run == 1 {
            println!("run 1/5 backend={backend:?} points={points} - ignore as warm up");
        } else {
            println!("run {run}/5 backend={backend:?} points={points}");
        }
        run_training::<NoopProfiler>(
            &backend,
            points,
            hidden_layer_size,
            learning_rate,
            iterations,
            None,
        )
    }
}

fn run_training<P: Profiler + Clone + std::fmt::Debug + 'static>(
    backend: &CliBackend,
    points: usize,
    hidden_layer_size: usize,
    learning_rate: f64,
    iterations: usize,
    profile_path: Option<&String>,
) {
    match backend {
        CliBackend::Seq => {
            let dataset = Dataset::circle(points);
            train::train::<CpuSeqBackend<P>, _>(
                dataset,
                learning_rate,
                iterations,
                hidden_layer_size,
                &mut TerseDebugger {},
                profile_path,
            );
        }
        CliBackend::Par => {
            let dataset = Dataset::circle(points);
            train::train::<CpuParBackend<P>, _>(
                dataset,
                learning_rate,
                iterations,
                hidden_layer_size,
                &mut TerseDebugger {},
                profile_path,
            );
        }
        CliBackend::Gpu => {
            let dataset = Dataset::circle(points);
            train::train::<GpuBackend<P>, _>(
                dataset,
                learning_rate as f32,
                iterations,
                hidden_layer_size,
                &mut TerseDebugger {},
                profile_path,
            );
        }
    }
}
