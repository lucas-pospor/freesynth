//! Times DIS on 100 different frame pairs, on the GPU or the CPU.
//!
//! ```text
//! cargo run --release -p freesynth-flow --example dis_bench -- FRAME_DIR [--preset P] [--runs N] [--pairs N]
//!     [--prop N] [--vr N] [--cpu [--threads N]]
//! ```
//!
//! All images in FRAME_DIR (sorted by name, same size) are uploaded once as 8-bit RGBA frames.
//! Pair `i` is frame `i mod n` with frame `(i mod n) + 1 + i / n` (mod n), so all pairs differ.
//!
//! - GPU time: every flow is one compute pass with timestamps at its start and end; all pairs
//!   are recorded into one command buffer and submitted at once. Reported per flow, as the
//!   median over pairs, and as the span from the first start to the last end divided by the
//!   number of pairs (which includes the gaps between passes).
//! - End to end: per pair, [`FlowEstimator::compute`] on grey frames in host memory (upload,
//!   compute, read back), timed on the host.
//!
//! - With `--cpu`, the CPU estimator instead: per pair, [`FlowEstimator::compute`] on grey frames
//!   in memory, timed on the host, on a thread pool of `--threads` threads (default: one per
//!   logical CPU).
//!
//! Each measurement is repeated `--runs` times (default 7) after two seconds of warm-up (GPU
//! clocks take most of a second to ramp up); the medians over the runs are printed. Host-timed
//! measurements give both the median and the mean time per flow of each run: the mean is what
//! sets the throughput of a long sequence. `--prop` and `--vr` override the preset's
//! propagation passes and refinement iterations.

use std::time::Instant;

use freesynth_flow::cpu::CpuDis;
use freesynth_flow::gpu::{FrameFormat, GpuContext, GpuDis, GpuFlow, GpuFrame};
use freesynth_flow::{DisParams, FlowEstimator, GrayImage, Preset};

/// Seconds of work before measuring.
const WARM_UP_SECONDS: f64 = 2.0;
/// Output buffers used in rotation.
const OUTPUTS: usize = 4;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut preset = Preset::Medium;
    let mut runs = 7usize;
    let mut pairs = 100usize;
    let (mut prop, mut vr) = (None, None);
    let (mut cpu, mut threads) = (false, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cpu" => cpu = true,
            "--threads" => threads = Some(args.next().ok_or("--threads needs a value")?.parse()?),
            "--prop" => prop = Some(args.next().ok_or("--prop needs a value")?.parse()?),
            "--vr" => vr = Some(args.next().ok_or("--vr needs a value")?.parse()?),
            "--preset" => preset = args.next().ok_or("--preset needs a value")?.parse()?,
            "--runs" => runs = args.next().ok_or("--runs needs a value")?.parse()?,
            "--pairs" => pairs = args.next().ok_or("--pairs needs a value")?.parse()?,
            _ => dir = Some(a),
        }
    }
    let dir = dir.ok_or("usage: dis_bench FRAME_DIR [--preset P] [--runs N] [--pairs N]")?;
    let mut paths: Vec<_> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    let frames: Vec<image::RgbaImage> =
        paths.iter().filter_map(|p| image::open(p).ok().map(|i| i.to_rgba8())).collect();
    if frames.len() < 2 {
        return Err("need at least two frames".into());
    }
    let (w, h) = (frames[0].width() as usize, frames[0].height() as usize);
    let n = frames.len();
    let pair = |i: usize| (i % n, (i % n + 1 + i / n) % n);
    let mut params = DisParams::preset(preset);
    params.propagation_passes = prop.unwrap_or(params.propagation_passes);
    params.vr_iterations = vr.unwrap_or(params.vr_iterations);
    if cpu {
        let gray: Vec<GrayImage> =
            frames.iter().map(|f| GrayImage::from_rgba8(w, h, f.as_raw())).collect::<Result<_, _>>()?;
        return bench_cpu(&gray, &params, pairs, runs, threads, pair);
    }

    let ctx = GpuContext::new()?;
    if let Some(name) = ctx.adapter_name() {
        println!("adapter: {name}");
    }
    if !ctx.has_timestamps() {
        return Err("the adapter does not support timestamp queries".into());
    }
    let dis = GpuDis::new(&ctx, w, h, &params)?;
    let plan = dis.plan();
    println!("{w}x{h}, {n} frames, {pairs} pairs, preset {preset}, levels {}..={}", plan.finest, plan.coarsest);

    let gpu_frames: Vec<GpuFrame> = frames
        .iter()
        .map(|f| {
            let g = GpuFrame::new(&ctx, w, h, FrameFormat::Rgba8);
            g.write_rgba8(&ctx, f.as_raw()).map(|_| g)
        })
        .collect::<Result<_, _>>()?;
    // A few output buffers in rotation: flows on one `GpuDis` run one after another anyway, and
    // a buffer per pair would take gigabytes at 4K.
    let outputs: Vec<GpuFlow> = (0..OUTPUTS).map(|_| GpuFlow::new(&ctx, w, h)).collect();
    let device = ctx.device();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("dis timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * pairs as u32,
    });
    let bytes = 16 * pairs as u64;
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps"),
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let period = f64::from(ctx.queue().get_timestamp_period()) * 1e-6; // ms per tick

    // One submission with all pairs; returns (median per-flow ms, span per flow ms).
    let batched = || -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut encoder = device.create_command_encoder(&Default::default());
        for i in 0..pairs {
            let (a, b) = pair(i);
            let out = &outputs[i % OUTPUTS];
            let ts = wgpu::ComputePassTimestampWrites {
                query_set: &queries,
                beginning_of_pass_write_index: Some(2 * i as u32),
                end_of_pass_write_index: Some(2 * i as u32 + 1),
            };
            dis.encode(&mut encoder, &gpu_frames[a], &gpu_frames[b], out, Some(ts))?;
        }
        encoder.resolve_query_set(&queries, 0..2 * pairs as u32, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, bytes);
        ctx.queue().submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map timestamps"));
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let ticks: Vec<u64> = bytemuck::cast_slice(&readback.get_mapped_range(..)?).to_vec();
        readback.unmap();
        let mut per: Vec<f64> = ticks.chunks(2).map(|t| t[1].wrapping_sub(t[0]) as f64 * period).collect();
        let span = ticks[2 * pairs - 1].wrapping_sub(ticks[0]) as f64 * period / pairs as f64;
        Ok((median(&mut per), span))
    };
    // GPU clocks ramp up under sustained load (it takes most of a second); warm up first.
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < WARM_UP_SECONDS {
        batched()?;
    }
    let (mut med, mut span): (Vec<f64>, Vec<f64>) =
        (0..runs).map(|_| batched()).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
    println!(
        "GPU time per flow (timestamps, one submission): median {:.3} ms, span/pairs {:.3} ms (medians of {runs} runs; run medians {:.3}-{:.3} ms)",
        median(&mut med),
        median(&mut span),
        med[0],
        med[runs - 1]
    );

    // End to end through `FlowEstimator::compute`: upload two grey frames, compute, read the
    // flow back, one pair at a time (grey conversion happens outside the timed region).
    let gray: Vec<GrayImage> =
        frames.iter().map(|f| GrayImage::from_rgba8(w, h, f.as_raw())).collect::<Result<_, _>>()?;
    let mut host_dis = GpuDis::new(&ctx, w, h, &params)?;
    let mut end_to_end = || -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut times = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let (a, b) = pair(i);
            let t = Instant::now();
            let flow = host_dis.compute(&gray[a], &gray[b])?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(flow);
        }
        let mean = times.iter().sum::<f64>() / pairs as f64;
        Ok((median(&mut times), mean))
    };
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < WARM_UP_SECONDS {
        end_to_end()?;
    }
    let (mut med, mut mean): (Vec<f64>, Vec<f64>) =
        (0..runs).map(|_| end_to_end()).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
    println!(
        "End to end per flow (upload + compute + readback): median {:.3} ms, mean {:.3} ms (medians of {runs} runs)",
        median(&mut med),
        median(&mut mean)
    );
    Ok(())
}

/// Host-timed CPU flows over `pairs` pairs, on a dedicated thread pool.
fn bench_cpu(
    gray: &[GrayImage],
    params: &DisParams,
    pairs: usize,
    runs: usize,
    threads: Option<usize>,
    pair: impl Fn(usize) -> (usize, usize) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = gray[0].size();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.unwrap_or(0)).build()?;
    let mut dis = CpuDis::new(w, h, params)?;
    let plan = dis.plan();
    println!(
        "{w}x{h}, {} frames, {pairs} pairs, CPU on {} threads, levels {}..={}",
        gray.len(),
        pool.current_num_threads(),
        plan.finest,
        plan.coarsest
    );
    pool.install(|| -> Result<(), freesynth_flow::FlowError> {
        let mut run = || -> Result<(f64, f64), freesynth_flow::FlowError> {
            let mut times = Vec::with_capacity(pairs);
            for i in 0..pairs {
                let (a, b) = pair(i);
                let t = Instant::now();
                let flow = dis.compute(&gray[a], &gray[b])?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(flow);
            }
            let mean = times.iter().sum::<f64>() / pairs as f64;
            Ok((median(&mut times), mean))
        };
        let warm = Instant::now();
        while warm.elapsed().as_secs_f64() < WARM_UP_SECONDS {
            run()?;
        }
        let (mut med, mut mean): (Vec<f64>, Vec<f64>) = (0..runs).map(|_| run()).collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
        let (lo, hi) = (med.iter().copied().fold(f64::INFINITY, f64::min), med.iter().copied().fold(0.0, f64::max));
        println!(
            "CPU time per flow: median {:.3} ms, mean {:.3} ms (medians of {runs} runs; run medians {lo:.3}-{hi:.3} ms)",
            median(&mut med),
            median(&mut mean)
        );
        Ok(())
    })?;
    Ok(())
}
