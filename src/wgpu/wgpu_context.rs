use std::{borrow::Cow, sync::LazyLock};

use wgpu::{
    Device, DeviceDescriptor, Features, Instance, Limits, MemoryHints, Queue, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, Trace,
};

// original version in kurtschelfthout/tensorken

#[derive(Debug)]
pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl WgpuContext {
    const MAP_SHADER: &'static str = include_str!("shaders/map.wgsl");

    const REPLACE_OP_NAME: &'static str = "replace_me_with_actual_operation";

    // neg, inv and relu are not supported out of the box
    const MAP_OPS: [&'static str; 7] = ["log", "exp", "sig", "id", "neg", "inv", "relu"];

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

    fn create_shader_module(&self, operation: &str, shader_source: &str) -> ShaderModule {
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(operation),
            source: ShaderSource::Wgsl(Cow::Borrowed(&shader_source)),
        })
    }
}

static WGPU_CONTEXT: LazyLock<WgpuContext> = LazyLock::new(|| WgpuContext::new());

pub fn get_wgpu_context() -> &'static WgpuContext {
    &WGPU_CONTEXT
}
