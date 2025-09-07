use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock, RwLockWriteGuard},
};

use wgpu::{
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance,
    Limits, MemoryHints, Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource, Trace,
};

// original version in kurtschelfthout/tensorken

#[derive(Debug)]
pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
    pipelines: RwLock<HashMap<&'static str, Arc<ComputePipeline>>>,
}

impl WgpuContext {
    const MAP_SHADER: &'static str = include_str!("shaders/map.wgsl");

    const REPLACE_OP_NAME: &'static str = "replace_me_with_actual_operation";

    // id, neg, inv and relu are not supported out of the box
    const MAP_OPS: [&'static str; 7] = ["log", "exp", "sig", "id", "neg", "inv", "relu"];

    const ENTRY_POINT: &'static str = "call";

    pub fn new() -> Self {
        let (device, queue) = Self::get_device_and_queue();
        Self {
            device,
            queue,
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_or_create_pipeline(&self, operation: &'static str) -> Option<Arc<ComputePipeline>> {
        self.pipelines
            .read()
            .unwrap()
            .get(operation)
            .map(Arc::clone)
            .or_else(|| {
                let module = if Self::MAP_OPS.contains(&operation) {
                    Some(self.create_shader_module(operation, Self::MAP_SHADER))
                } else {
                    None
                };

                module.and_then(|m| {
                    let mut pipelines = self.pipelines.write().unwrap();
                    self.insert_pipeline(operation, &m, &mut pipelines);
                    pipelines.get(&operation).map(Arc::clone)
                })
            })
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

    fn insert_pipeline(
        &self,
        operation: &'static str,
        module: &ShaderModule,
        pipelines: &mut RwLockWriteGuard<HashMap<&'static str, Arc<ComputePipeline>>>,
    ) {
        let compute_pipeline = Arc::new(self.device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label: Some(operation),
                layout: None,
                module,
                entry_point: Some(Self::ENTRY_POINT),
                cache: None,
                compilation_options: Default::default(),
            },
        ));
        pipelines.insert(operation, compute_pipeline);
    }
}

static WGPU_CONTEXT: LazyLock<WgpuContext> = LazyLock::new(|| WgpuContext::new());

pub fn get_wgpu_context() -> &'static WgpuContext {
    &WGPU_CONTEXT
}
