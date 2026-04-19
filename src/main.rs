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
    #[arg(value_enum, short, long, default_value_t = CliBackend::Seq)]
    backend: CliBackend,
    #[arg(short, long, default_value_t = 4)]
    power_ten_points: u32,
    #[arg(short, long, default_value_t = 500)]
    iterations: usize,
    #[arg(short, long, default_value_t = 0.1)]
    learning_rate: f64,
    #[arg(long, default_value_t = 50)]
    hidden_layer_size: usize,
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand)]
enum CliCommand {
    Viz,
    Benchmark {
        #[arg(short, long)]
        seed: Option<u64>,
    },
    Profile {
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
    let points = 10_usize.pow(cli.power_ten_points);

    match cli.command {
        Some(CliCommand::Benchmark { seed }) => {
            for run in 1..=5 {
                if run == 1 {
                    println!(
                        "run 1/5 backend={:?} points={points} - ignore as warm up",
                        cli.backend
                    );
                } else {
                    println!("run {run}/5 backend={:?} points={points}", cli.backend);
                }
                run_training::<NoopProfiler>(
                    &cli.backend,
                    points,
                    cli.hidden_layer_size,
                    cli.learning_rate,
                    cli.iterations,
                    seed,
                    None,
                );
            }
            Ok(())
        }
        Some(CliCommand::Profile { seed, output_path }) => {
            run_training::<CsvProfiler>(
                &cli.backend,
                points,
                cli.hidden_layer_size,
                cli.learning_rate,
                cli.iterations,
                seed,
                Some(&output_path),
            );
            Ok(())
        }
        Some(CliCommand::Viz) => viz(points, cli),
        None => {
            run_training::<NoopProfiler>(
                &cli.backend,
                points,
                cli.hidden_layer_size,
                cli.learning_rate,
                cli.iterations,
                None,
                None,
            );
            Ok(())
        }
    }
}

fn viz(points: usize, cli: Cli) -> Result<(), Error> {
    let dataset = Dataset::circle(points, None);
    let mut debugger = VizDebugger::new(&dataset, cli.iterations);
    let mut debugger_thread_clone = debugger.clone();

    match cli.backend {
        CliBackend::Seq => {
            std::thread::spawn(move || {
                train::train::<CpuSeqBackend, _>(
                    dataset,
                    cli.learning_rate,
                    cli.iterations,
                    cli.hidden_layer_size,
                    &mut debugger_thread_clone,
                    None,
                );
            });
        }
        CliBackend::Par => {
            std::thread::spawn(move || {
                train::train::<CpuParBackend, _>(
                    dataset,
                    cli.learning_rate,
                    cli.iterations,
                    cli.hidden_layer_size,
                    &mut debugger_thread_clone,
                    None,
                );
            });
        }
        #[cfg(feature = "gpu")]
        CliBackend::Gpu => {
            std::thread::spawn(move || {
                train::train::<GpuBackend, _>(
                    dataset.to_f32(),
                    cli.learning_rate as f32,
                    cli.iterations,
                    cli.hidden_layer_size,
                    &mut debugger_thread_clone,
                    None,
                );
            });
        }
    }

    debugger.run()
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
    let dataset = Dataset::circle(points, seed);
    match backend {
        CliBackend::Seq => {
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
            train::train::<GpuBackend<P>, _>(
                dataset.to_f32(),
                learning_rate as f32,
                iterations,
                hidden_layer_size,
                &mut TerseDebugger {},
                profile_path,
            );
        }
    }
}
