use std::io::Error;

#[cfg(feature = "gpu")]
use allumette::backend::backend::GpuBackend;
use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend},
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
        #[arg(short, long)]
        seed: Option<u64>,
    },
    Profile {
        #[arg(value_enum, short, long)]
        backend: CliBackend,
        #[arg(short, long, default_value_t = 4)]
        power_ten_points: u32,
        #[arg(short, long)]
        seed: Option<u64>,
        #[arg(short, long)]
        output_path: String,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum CliBackend {
    Seq,
    Par,
    #[cfg(feature = "gpu")]
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
            seed,
        }) => {
            benchmark(
                backend,
                power_ten_points,
                hidden_layer_size,
                learning_rate,
                iterations,
                seed,
            );
            Ok(())
        }
        Some(CliCommand::Profile {
            backend,
            power_ten_points,
            seed,
            output_path,
        }) => {
            profile(
                backend,
                power_ten_points,
                hidden_layer_size,
                learning_rate,
                iterations,
                seed,
                output_path,
            );
            Ok(())
        }
        _ => viz(hidden_layer_size, learning_rate, iterations),
    }
}

fn viz(hidden_layer_size: usize, learning_rate: f64, iterations: usize) -> Result<(), Error> {
    let pts = 1000;
    let dataset = Dataset::circle(pts, None);

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
    seed: Option<u64>,
    output_path: String,
) {
    let points = 10_usize.pow(power_ten_points);
    run_training::<CsvProfiler>(
        &backend,
        points,
        hidden_layer_size,
        learning_rate,
        iterations,
        seed,
        Some(&output_path),
    )
}

fn benchmark(
    backend: CliBackend,
    power_ten_points: u32,
    hidden_layer_size: usize,
    learning_rate: f64,
    iterations: usize,
    seed: Option<u64>,
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
            seed,
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
    seed: Option<u64>,
    profile_path: Option<&String>,
) {
    match backend {
        CliBackend::Seq => {
            let dataset = Dataset::circle(points, seed);
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
            let dataset = Dataset::circle(points, seed);
            train::train::<CpuParBackend<P>, _>(
                dataset,
                learning_rate,
                iterations,
                hidden_layer_size,
                &mut TerseDebugger {},
                profile_path,
            );
        }
        #[cfg(feature = "gpu")]
        CliBackend::Gpu => {
            let dataset = Dataset::circle(points, seed);
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
