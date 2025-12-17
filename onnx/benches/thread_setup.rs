use std::sync::Once;

// Initialize a global Rayon thread pool and configure each worker thread:
// - Set thread priority to TimeCritical
// - Pin each worker to a specific logical core (best-effort)
// Safe to call multiple times; only the first call will take effect.
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_impl();
    });
}

fn init_impl() {
    // Defer imports to avoid forcing dependencies when not used by a bench.
    use gdt_cpus::{ThreadPriority, num_logical_cores, pin_thread_to_core, set_thread_priority};

    // Determine number of logical cores for pinning policy.
    // gdt_cpus returns Result<usize, Error>; fall back to 1 on error.
    let cores = num_logical_cores().unwrap_or(1);

    // Build the global rayon pool with start handler to set priority and affinity.
    // If the global pool is already built, ignore the error (can't reconfigure).
    let _ = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rayon-w{}", i))
        .start_handler(move |index| {
            // Best-effort: set time-critical priority first for lower latency.
            let _ = set_thread_priority(ThreadPriority::TimeCritical);

            // Pin worker to a logical core in a round-robin fashion.
            // On platforms where affinity isn't supported (e.g., Apple Silicon), this will be a no-op.
            let core_id = index % cores;
            let _ = pin_thread_to_core(core_id);
        })
        .build_global();
}
