use std::io::Error;

#[cfg(feature = "gpu")]
use allumette::backend::backend::GpuBackend;
use allumette::{
    backend::backend::{CpuParBackend, CpuSeqBackend},
    math::element::Element,
    training::{dataset::Dataset, train},
    util::{
        debugger::{TerseDebugger, VizDebugger},
        profiler::{CsvProfiler, NoopProfiler, Profiler},
        unsafe_usize_convert::UnsafeUsizeConvert,
    },
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(value_enum, short, long, default_value_t = CliBackend::Seq)]
    backend: CliBackend,
    #[arg(value_enum, short, long, default_value_t = CliDataset::Star)]
    dataset: CliDataset,
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

#[derive(ValueEnum, Clone, Debug)]
enum CliDataset {
    Simple,
    Diag,
    Split,
    Xor,
    Circle,
    Star,
}

impl CliDataset {
    fn create<E: Element + UnsafeUsizeConvert>(
        &self,
        points: usize,
        seed: Option<u64>,
    ) -> Dataset<E> {
        match self {
            CliDataset::Simple => Dataset::simple(points, seed),
            CliDataset::Star => Dataset::star(points, seed),
            CliDataset::Xor => Dataset::xor(points, seed),
            CliDataset::Circle => Dataset::circle(points, seed),
            CliDataset::Diag => Dataset::circle(points, seed),
            CliDataset::Split => Dataset::split(points, seed),
        }
    }
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();

    match cli.command {
        Some(CliCommand::Benchmark { seed }) => {
            for run in 1..=5 {
                if run == 1 {
                    println!("run 1/5 backend={:?} - ignore as warm up", cli.backend);
                } else {
                    println!("run {run}/5 backend={:?}", cli.backend);
                }
                run_training::<NoopProfiler>(&cli, seed, None);
            }
            Ok(())
        }
        Some(CliCommand::Profile {
            seed,
            ref output_path,
        }) => {
            run_training::<CsvProfiler>(&cli, seed, Some(output_path));
            Ok(())
        }
        Some(CliCommand::Viz) => viz(&cli),
        None => {
            run_training::<NoopProfiler>(&cli, None, None);
            Ok(())
        }
    }
}

fn viz(cli: &Cli) -> Result<(), Error> {
    let points = 10_usize.pow(cli.power_ten_points);
    let dataset = cli.dataset.create(points, None);
    let lr = cli.learning_rate;
    let its = cli.iterations;
    let hls = cli.hidden_layer_size;
    let mut debugger = VizDebugger::new(&dataset, cli.iterations);
    let mut debugger_thread_clone = debugger.clone();

    match cli.backend {
        CliBackend::Seq => {
            std::thread::spawn(move || {
                train::train::<CpuSeqBackend, _>(
                    dataset,
                    lr,
                    its,
                    hls,
                    &mut debugger_thread_clone,
                    None,
                );
            });
        }
        CliBackend::Par => {
            std::thread::spawn(move || {
                train::train::<CpuParBackend, _>(
                    dataset,
                    lr,
                    its,
                    hls,
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
                    lr as f32,
                    its,
                    hls,
                    &mut debugger_thread_clone,
                    None,
                );
            });
        }
    }

    debugger.run()
}

fn run_training<P: Profiler + Clone + std::fmt::Debug + 'static>(
    cli: &Cli,
    seed: Option<u64>,
    output_path: Option<&String>,
) {
    let points = 10_usize.pow(cli.power_ten_points);
    let dataset = cli.dataset.create(points, seed);
    match cli.backend {
        CliBackend::Seq => {
            train::train::<CpuSeqBackend<P>, _>(
                dataset,
                cli.learning_rate,
                cli.iterations,
                cli.hidden_layer_size,
                &mut TerseDebugger {},
                output_path,
            );
        }
        CliBackend::Par => {
            train::train::<CpuParBackend<P>, _>(
                dataset,
                cli.learning_rate,
                cli.iterations,
                cli.hidden_layer_size,
                &mut TerseDebugger {},
                output_path,
            );
        }
        #[cfg(feature = "gpu")]
        CliBackend::Gpu => {
            train::train::<GpuBackend<P>, _>(
                dataset.to_f32(),
                cli.learning_rate as f32,
                cli.iterations,
                cli.hidden_layer_size,
                &mut TerseDebugger {},
                output_path,
            );
        }
    }
}
