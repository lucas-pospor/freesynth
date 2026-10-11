//! GPU backend: DIS in WGSL compute shaders on wgpu.
//!
//! [`GpuContext`] holds the device. It can open its own device ([`GpuContext::new`]) or wrap a
//! device the caller already has ([`GpuContext::from_device`]), so that flow, synthesis and a
//! user interface can share one device and pass buffers without copying through the host.
//!
//! [`GpuDis`] computes flow between two frames. The simple path is
//! [`FlowEstimator::compute`](crate::FlowEstimator::compute) (upload, compute, read back). To
//! keep everything on the GPU, put frames in [`GpuFrame`]s, record flows with
//! [`GpuDis::encode`] into your own command encoder (as many as you like per submission), and
//! read the resulting [`GpuFlow`] buffers in your own kernels.

mod dis;

use std::sync::mpsc;

use crate::error::{FlowError, invalid};
use crate::flo::FlowField;
use crate::image::GrayImage;

pub use self::dis::GpuDis;

/// A wgpu device and queue used by the flow kernels.
#[derive(Clone, Debug)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: Option<wgpu::AdapterInfo>,
}

impl GpuContext {
    /// Opens the default (high-performance) adapter. Timestamp queries are enabled when the
    /// adapter supports them, and buffer size limits are raised to the adapter's maximum.
    pub fn new() -> Result<Self, FlowError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| FlowError::NoAdapter)?;
        let info = adapter.get_info();
        let available = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_buffer_size: available.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("freesynth-flow"),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| FlowError::Device(e.to_string()))?;
        Ok(Self { device, queue, adapter: Some(info) })
    }

    /// Wraps an existing device and queue. The device needs at least the default limits;
    /// [`GpuDis::new`] reports an error when a frame size exceeds its buffer limits.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue, adapter: None }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Description of the adapter, when this context opened it.
    pub fn adapter_name(&self) -> Option<String> {
        self.adapter.as_ref().map(|i| format!("{} ({:?}, {:?}, driver {})", i.name, i.device_type, i.backend, i.driver_info))
    }

    /// True when the device can write timestamps at compute pass boundaries.
    pub fn has_timestamps(&self) -> bool {
        self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Runs `f`, turning wgpu validation and out-of-memory errors raised meanwhile into
    /// [`FlowError::Gpu`].
    pub(crate) fn scoped<T>(&self, f: impl FnOnce() -> Result<T, FlowError>) -> Result<T, FlowError> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = f();
        let validation = pollster::block_on(validation.pop());
        let oom = pollster::block_on(oom.pop());
        if let Some(e) = validation.or(oom) {
            return Err(FlowError::Gpu(e.to_string()));
        }
        result
    }

    /// Waits for all submitted work.
    pub(crate) fn wait(&self) -> Result<(), FlowError> {
        self.device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| FlowError::Gpu(e.to_string()))
    }
}

/// Pixel format of a [`GpuFrame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFormat {
    /// One `f32` intensity per pixel (0-255 for the default parameters).
    Gray32F,
    /// One `u32` per pixel holding 8-bit RGBA, red in the lowest byte (the byte order of
    /// interleaved RGBA8 data). Converted to Rec. 601 luma on the GPU.
    Rgba8,
}

/// A frame in a GPU storage buffer, 4 bytes per pixel, row-major.
#[derive(Clone, Debug)]
pub struct GpuFrame {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
    format: FrameFormat,
}

impl GpuFrame {
    /// Allocates an uninitialised frame.
    pub fn new(ctx: &GpuContext, width: usize, height: usize, format: FrameFormat) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow frame"),
            size: (width * height * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, width, height, format }
    }

    /// Uses an existing buffer, which must have `STORAGE` usage and hold at least
    /// `width * height` 4-byte pixels.
    pub fn from_buffer(buffer: wgpu::Buffer, width: usize, height: usize, format: FrameFormat) -> Result<Self, FlowError> {
        if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return invalid("frame buffers need STORAGE usage");
        }
        if buffer.size() < (width * height * 4) as u64 {
            return invalid(format!("buffer of {} bytes is too small for a {width}x{height} frame", buffer.size()));
        }
        Ok(Self { buffer, width, height, format })
    }

    /// Writes an intensity image into a [`FrameFormat::Gray32F`] frame (through the queue).
    pub fn write_gray(&self, ctx: &GpuContext, image: &GrayImage) -> Result<(), FlowError> {
        if self.format != FrameFormat::Gray32F || image.size() != (self.width, self.height) {
            return invalid("image does not match the frame's size or format");
        }
        ctx.queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(image.data()));
        Ok(())
    }

    /// Writes interleaved 8-bit RGBA into a [`FrameFormat::Rgba8`] frame (through the queue).
    pub fn write_rgba8(&self, ctx: &GpuContext, rgba: &[u8]) -> Result<(), FlowError> {
        if self.format != FrameFormat::Rgba8 || rgba.len() != self.width * self.height * 4 {
            return invalid("data does not match the frame's size or format");
        }
        ctx.queue.write_buffer(&self.buffer, 0, rgba);
        Ok(())
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn format(&self) -> FrameFormat {
        self.format
    }
}

/// A flow field in a GPU storage buffer: one `vec2<f32>` per pixel, row-major, in the
/// convention of [`FlowField`].
#[derive(Clone, Debug)]
pub struct GpuFlow {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
}

impl GpuFlow {
    /// Allocates an uninitialised flow buffer.
    pub fn new(ctx: &GpuContext, width: usize, height: usize) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow field"),
            size: (width * height * 8).max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, width, height }
    }

    /// Uses an existing buffer, which must have `STORAGE` usage and hold at least
    /// `width * height * 8` bytes.
    pub fn from_buffer(buffer: wgpu::Buffer, width: usize, height: usize) -> Result<Self, FlowError> {
        if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return invalid("flow buffers need STORAGE usage");
        }
        if buffer.size() < (width * height * 8) as u64 {
            return invalid(format!("buffer of {} bytes is too small for a {width}x{height} flow", buffer.size()));
        }
        Ok(Self { buffer, width, height })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Copies the flow to the host, waiting for all submitted work. The buffer needs
    /// `COPY_SRC` usage (buffers from [`GpuFlow::new`] have it).
    pub fn read(&self, ctx: &GpuContext) -> Result<FlowField, FlowError> {
        let staging = Staging::new(ctx, self.width, self.height);
        staging.read(ctx, self)
    }
}

/// A mappable buffer for reading flow fields back.
#[derive(Debug)]
pub(crate) struct Staging {
    buffer: wgpu::Buffer,
}

impl Staging {
    pub(crate) fn new(ctx: &GpuContext, width: usize, height: usize) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow readback"),
            size: (width * height * 8).max(8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    /// Copies `flow` into the staging buffer, submits, waits and converts.
    pub(crate) fn read(&self, ctx: &GpuContext, flow: &GpuFlow) -> Result<FlowField, FlowError> {
        let bytes = (flow.width * flow.height * 8) as u64;
        if bytes > self.buffer.size() {
            return invalid("staging buffer is too small");
        }
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("flow readback") });
        if bytes > 0 {
            encoder.copy_buffer_to_buffer(&flow.buffer, 0, &self.buffer, 0, bytes);
        }
        ctx.queue.submit([encoder.finish()]);
        if bytes == 0 {
            return Ok(FlowField::new(flow.width, flow.height));
        }
        let (tx, rx) = mpsc::channel();
        self.buffer.map_async(wgpu::MapMode::Read, ..bytes, move |r| {
            let _ = tx.send(r);
        });
        ctx.wait()?;
        rx.recv()
            .map_err(|_| FlowError::Gpu("buffer mapping was cancelled".into()))?
            .map_err(|e| FlowError::Gpu(format!("cannot read back the flow: {e}")))?;
        let data = {
            let view = self.buffer.get_mapped_range(..bytes).map_err(|e| FlowError::Gpu(e.to_string()))?;
            bytemuck::cast_slice::<u8, [f32; 2]>(&view).to_vec()
        };
        self.buffer.unmap();
        Ok(FlowField::from_vec(flow.width, flow.height, data))
    }
}
