use std::path::PathBuf;
use std::sync::{Arc, Once};
use std::time::Instant;

use par_iter::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

// Use snmalloc for improved allocation performance, like in the benchmark.
#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

// Initialize a global Rayon thread pool and configure each worker thread.
fn init_thread_pool() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        use gdt_cpus::{
            ThreadPriority, num_logical_cores, pin_thread_to_core, set_thread_priority,
        };

        let cores = num_logical_cores().unwrap_or(1);

        let _ = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("rayon-w{}", i))
            .start_handler(move |index| {
                let _ = set_thread_priority(ThreadPriority::TimeCritical);
                let core_id = index % cores;
                let _ = pin_thread_to_core(core_id);
            })
            .build_global();
    });
}

fn main() {
    // Ensure Rayon worker threads are configured before any parallel work.
    init_thread_pool();

    // Configuration via environment variables (same as the benchmark).
    // Required:
    //   MODEL_PATH: path to an .onnx containing ai.onnx.ml.LinearClassifier
    //   INPUT: input feature dimension (usize)
    //   OUTPUT: number of classes (usize)
    // Optional:
    //   SAMPLES: number of input samples to process in parallel (default 15000)
    //   STEPS: number of repeated runs per sample within the state (default 2800)

    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH to an existing .onnx file containing ai.onnx.ml.LinearClassifier");
    let onnx_path = PathBuf::from(&model_path);
    assert!(onnx_path.exists(), "Model path does not exist: {}", model_path);

    let input_dim: usize = std::env::var("INPUT")
        .expect("Set INPUT to the input feature dimension (usize)")
        .parse()
        .expect("INPUT must be a positive integer");
    let output_dim: usize = std::env::var("OUTPUT")
        .expect("Set OUTPUT to the number of classes (usize)")
        .parse()
        .expect("OUTPUT must be a positive integer");

    let samples: usize =
        std::env::var("SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(15_000);
    let steps: usize = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2_800);

    eprintln!(
        "Config:\n  MODEL_PATH = {}\n  INPUT = {}\n  OUTPUT = {}\n  SAMPLES = {}\n  STEPS = {}",
        onnx_path.display(),
        input_dim,
        output_dim,
        samples,
        steps
    );

    // Load model once for shape inference
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();
    let n = model.sym("N");

    // Configure input and output dimensions
    let model = model
        .with_input_fact(0, f32::fact(dims!(n, input_dim)).into())
        .unwrap()
        .with_output_fact(0, i64::fact(dims!(n)).into())
        .unwrap()
        .with_output_fact(1, f32::fact(dims!(n, output_dim)).into())
        .unwrap()
        .into_optimized()
        .unwrap();

    let input_fact = model.input_fact(0).unwrap().clone();
    let shape: TVec<usize> = input_fact
        .shape
        .as_concrete()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_else(|| tvec![1, input_dim]);
    let num_features = shape[1];

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let input_tensors: Arc<Vec<Tensor>> = Arc::new(
        (0..samples)
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.gen_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    let plan = TypedSimplePlan::new(model.clone()).unwrap();

    let start = Instant::now();
    (0..samples).into_par_iter().for_each(|i| {
        let input_val = input_tensors[i].clone().into_tvalue();
        let mut state = SimpleState::new(plan.clone()).unwrap();
        let mut inputs = TVec::new();

        for _ in 0..steps {
            inputs.clear();
            inputs.push(input_val.clone());
            let _ = state.run(inputs.clone()).unwrap();
        }
    });
    let duration = start.elapsed();

    println!("Completed parallel run in {:?}", duration);
}
