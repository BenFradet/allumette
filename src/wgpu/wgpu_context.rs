use std::sync::Once;

use wgpu::{Device, DeviceDescriptor, Features, Instance, Limits, MemoryHints, Queue, Trace};

// taken from kurtschelfthout/tensorken

#[derive(Debug)]
pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl WgpuContext {
    fn new() -> Self {
        let (device, queue) = Self::get_device_and_queue();
        Self { device, queue }
    }

    async fn get_device_and_queue_async() -> (Device, Queue) {
        let instance = Instance::default();
        let adapter = wgpu::util::initialize_adapter_from_env(&instance, None)
            .expect("No suitable GPU adapters found on the system");
        let info = adapter.get_info();
        println!(
            "Using {:#?} {} with {:#?} backend",
            info.device_type, info.name, info.backend
        );
        let device_and_queue = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: Features::empty(),
                required_limits: Limits::downlevel_defaults(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .unwrap();
        device_and_queue
    }

    fn get_device_and_queue() -> (Device, Queue) {
        futures::executor::block_on(Self::get_device_and_queue_async())
    }
}

static mut WGPU_CONTEXT: Option<WgpuContext> = None;
static INIT_WGPU_CONTEXT: Once = Once::new();

pub fn get_wgpu_context() -> &'static WgpuContext {
    unsafe {
        INIT_WGPU_CONTEXT.call_once(|| WGPU_CONTEXT = Some(WgpuContext::new()));
        WGPU_CONTEXT.as_ref().unwrap()
    }
}
