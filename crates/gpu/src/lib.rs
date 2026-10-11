//! wgpu compute backend.

/// Human-readable descriptions of all GPU adapters wgpu can see.
pub fn list_adapters() -> Vec<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| {
            let i = a.get_info();
            format!("{} ({:?}, {:?}, driver {})", i.name, i.device_type, i.backend, i.driver_info)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn lists_adapters() {
        let adapters = super::list_adapters();
        for a in &adapters {
            println!("{a}");
        }
        assert!(!adapters.is_empty(), "no GPU adapters found");
    }
}
