use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock, RwLockWriteGuard},
};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupLayout, Buffer, BufferDescriptor, BufferUsages, CommandBuffer,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Features, Instance, Limits, MemoryHints, PipelineLayout,
    PipelineLayoutDescriptor, PollError, PollStatus, PollType, Queue, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, Trace,
};

use crate::{shaping::iter::Iter, wgpu::workgroup_info::WorkgroupInfo};

pub const WGPU_ELEMENT_SIZE: usize = std::mem::size_of::<f32>();

// inspired from kurtschelfthout/tensorken
#[derive(Debug)]
pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
    pipelines: RwLock<HashMap<&'static str, Arc<ComputePipeline>>>,
}

impl WgpuContext {
    const MAP_SHADER: &'static str = include_str!("shaders/map.wgsl");

    const REPLACE_OP_NAME: &'static str = "replace_with_actual_operation";
    const REPLACE_WORKGROUP_SIZE: &'static str = "@workgroup_size(1)";

    // id, neg, inv and relu are not supported out of the box
    const MAP_OPS: [&'static str; 7] = ["log", "exp", "sig", "id", "neg", "inv", "relu"];

    const ENTRY_POINT: &'static str = "call";

    pub fn new() -> Self {
        let (device, queue) = Self::get_device_and_queue();
        unsafe {
            device.start_graphics_debugger_capture();
        }
        Self {
            device,
            queue,
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    pub fn encode_command(
        &self,
        workgroup_info: &WorkgroupInfo,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
    ) -> CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, Some(bind_group), &[]);
            compute_pass.dispatch_workgroups(workgroup_info.count.try_into().unwrap(), 1, 1);
        }
        encoder.finish()
    }

    // blocks until execution is complete
    pub fn submit_command(&self, command_buffer: CommandBuffer) -> Result<PollStatus, PollError> {
        let index = self.queue.submit(Some(command_buffer));
        self.device.poll(PollType::WaitForSubmissionIndex(index))
    }

    pub fn create_pipeline_layout(&self, bind_group_layout: &BindGroupLayout) -> PipelineLayout {
        self.device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("pipeline layout"),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            })
    }

    pub fn get_or_create_pipeline(
        &self,
        operation: &'static str,
        workgroup_info: &WorkgroupInfo,
        pipeline_layout: &PipelineLayout,
    ) -> Option<Arc<ComputePipeline>> {
        self.pipelines
            .read()
            .unwrap()
            .get(operation)
            .map(Arc::clone)
            .or_else(|| {
                let module = if Self::MAP_OPS.contains(&operation) {
                    Some(self.create_shader_module(
                        operation,
                        &Self::MAP_SHADER.replace(Self::REPLACE_OP_NAME, operation),
                        &workgroup_info,
                    ))
                } else {
                    None
                };
                println!("module created {:?}", module);

                module.and_then(|m| {
                    let mut pipelines = self.pipelines.write().unwrap();
                    self.insert_pipeline(operation, &m, &pipeline_layout, &mut pipelines);
                    println!("pipeline inserted");
                    pipelines.get(&operation).map(Arc::clone)
                })
            })
    }

    pub fn create_output_buffer(&self, size: u64, operation: &str, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Tensor {operation}")),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn create_storage_buffer(&self, iter: Iter<'_>, label: &str) -> Buffer {
        let data: Vec<_> = iter.map(|u| u32::try_from(u).unwrap()).collect();
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE,
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
        println!("GPU limits: {:#?}", adapter.limits());
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            panic!("Adapter does not support compute shaders");
        }
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

    fn create_shader_module(
        &self,
        operation: &str,
        shader_source: &str,
        workgroup_info: &WorkgroupInfo,
    ) -> ShaderModule {
        let source = shader_source.replace(
            Self::REPLACE_WORKGROUP_SIZE,
            &workgroup_info.workgroup_size(),
        );
        println!("shader source: {:?}", source);
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(operation),
            source: ShaderSource::Wgsl(Cow::Borrowed(&source)),
        })
    }

    fn insert_pipeline(
        &self,
        operation: &'static str,
        module: &ShaderModule,
        pipeline_layout: &PipelineLayout,
        pipelines: &mut RwLockWriteGuard<HashMap<&'static str, Arc<ComputePipeline>>>,
    ) {
        let compute_pipeline = Arc::new(self.device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label: Some(operation),
                layout: Some(pipeline_layout),
                module,
                entry_point: Some(Self::ENTRY_POINT),
                cache: None,
                compilation_options: Default::default(),
            },
        ));
        println!("compute pipeline created: {:?}", compute_pipeline);
        pipelines.insert(operation, compute_pipeline);
    }
}

static WGPU_CONTEXT: LazyLock<WgpuContext> = LazyLock::new(|| WgpuContext::new());

pub fn get_wgpu_context() -> &'static WgpuContext {
    &WGPU_CONTEXT
}
