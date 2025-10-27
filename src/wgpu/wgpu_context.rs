use std::{
    borrow::Cow,
    collections::HashMap,
    num::NonZeroU64,
    sync::{Arc, LazyLock, RwLock, RwLockWriteGuard},
};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandBuffer, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    Limits, MemoryHints, PipelineLayout, PipelineLayoutDescriptor, PollError, PollStatus, PollType,
    Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, Trace,
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
            let x = workgroup_info.count.try_into().unwrap();
            println!("x {x:?}");
            compute_pass.dispatch_workgroups(x, 1, 1);
        }
        encoder.finish()
    }

    // blocks until execution is complete
    pub fn submit_command(&self, command_buffer: CommandBuffer) -> Result<PollStatus, PollError> {
        let index = self.queue.submit([command_buffer]);
        self.device.poll(PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
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
    ) -> Option<Arc<ComputePipeline>> {
        let pipeline_opt = {
            let pipelines = self.pipelines.read().unwrap();
            pipelines.get(operation).map(Arc::clone)
        };

        pipeline_opt.or_else(|| {
            let module = if Self::MAP_OPS.contains(&operation) {
                Some(self.create_shader_module(
                    operation,
                    &Self::MAP_SHADER.replace(Self::REPLACE_OP_NAME, operation),
                    &workgroup_info,
                ))
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

    pub fn create_bind_group(
        &self,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        metadata_buffer: &Buffer,
        bind_group_layout: &BindGroupLayout,
    ) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("map bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: metadata_buffer.as_entire_binding(),
                },
            ],
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
                experimental_features: ExperimentalFeatures::disabled(),
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
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(operation),
            source: ShaderSource::Wgsl(Cow::Borrowed(&source)),
        })
    }

    // TODO: change based on shader
    fn create_bind_group_layout(&self) -> BindGroupLayout {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                ],
            })
    }

    fn insert_pipeline(
        &self,
        operation: &'static str,
        module: &ShaderModule,
        pipelines: &mut RwLockWriteGuard<HashMap<&'static str, Arc<ComputePipeline>>>,
    ) {
        let bind_group_layout = self.create_bind_group_layout();
        let pipeline_layout = self.create_pipeline_layout(&bind_group_layout);
        let compute_pipeline = Arc::new(self.device.create_compute_pipeline(
            &ComputePipelineDescriptor {
                label: Some(operation),
                layout: Some(&pipeline_layout),
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
