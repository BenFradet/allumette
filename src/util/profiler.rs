use std::{sync::Mutex, time::Instant};

pub trait Profiler {
    const ENABLED: bool;
    fn start() -> Self;
    fn stop(self, tag: &'static str);
    fn flush(path: &str);
}

#[derive(Clone, Debug)]
pub struct NoopProfiler;
impl Profiler for NoopProfiler {
    const ENABLED: bool = false;
    fn start() -> Self {
        Self
    }
    fn stop(self, _tag: &'static str) {}
    fn flush(_path: &str) {}
}

#[derive(Clone, Debug)]
pub struct CsvProfiler {
    start: Instant,
}

static ENTRIES: Mutex<Vec<(&'static str, u128)>> = Mutex::new(vec![]);

impl Profiler for CsvProfiler {
    const ENABLED: bool = true;

    fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    fn stop(self, tag: &'static str) {
        let dur = self.start.elapsed().as_micros();
        ENTRIES.lock().unwrap().push((tag, dur));
    }

    fn flush(path: &str) {
        let entries = ENTRIES.lock().unwrap();
        let mut out = String::from("op,duration_micros\n");
        for (tag, micros) in entries.iter() {
            out.push_str(&format!("{tag},{micros}\n"));
        }
        std::fs::write(path, out).expect("failed to write csv profiling info");
    }
}
