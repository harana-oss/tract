use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use par_iter::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

mod thread_setup;

#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

fn bench_linear_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_linear_classifier");
    group.sample_size(10);

    // Ensure Rayon worker threads are configured before any parallel work.
    //thread_setup::init();

    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH to an existing .onnx file containing ai.onnx.ml.LinearClassifier");
    let onnx_path = PathBuf::from(&model_path);
    assert!(onnx_path.exists(), "Model path does not exist: {}", model_path);

    // Load model once for shape inference
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();

    // Configure input and output dimensions
    let input_dim: usize = std::env::var("INPUT")
        .expect("Set INPUT to the input feature dimension (usize)")
        .parse()
        .expect("INPUT must be a positive integer");
    let output_dim: usize = std::env::var("OUTPUT")
        .expect("Set OUTPUT to the number of classes (usize)")
        .parse()
        .expect("OUTPUT must be a positive integer");

    let model = model
        .with_input_fact(0, f32::fact(dims!(1, input_dim)).into())
        .unwrap()
        .with_output_fact(0, i64::fact(dims!(1)).into())
        .unwrap()
        .with_output_fact(1, f32::fact(dims!(1, output_dim)).into())
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
        (0..(15_000))
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.gen_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    let plan = TypedSimplePlan::new(model.clone()).unwrap();

    group.bench_function(
        BenchmarkId::new("load_opt_run_parallel", onnx_path.display().to_string()),
        |b| {
            let tensors = Arc::clone(&input_tensors);

            b.iter_custom(|_| {
                let start = Instant::now();
                // Optional Rayon split tuning via env vars PAR_MIN_LEN / PAR_MAX_LEN
                let (min_len, max_len) = {
                    use std::env;
                    let mut min = env::var("PAR_MIN_LEN")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(1);
                    let mut max = env::var("PAR_MAX_LEN")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(usize::MAX);
                    if min == 0 {
                        min = 1;
                    }
                    if max == 0 {
                        max = 1;
                    }
                    if min > max {
                        std::mem::swap(&mut min, &mut max);
                    }
                    (min, max)
                };

                (0..15_000).into_par_iter().with_min_len(min_len).with_max_len(max_len).for_each(
                    |i| {
                        let input_val = tensors[i].clone().into_tvalue();
                        let mut state = SimpleState::new(plan.clone()).unwrap();
                        let mut inputs = TVec::new();

                        for _ in 0..2800 {
                            inputs.clear();
                            inputs.push(input_val.clone());
                            let _ = state.run(inputs.clone()).unwrap();
                        }
                    },
                );

                start.elapsed()
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_linear_classifier);
criterion_main!(benches);
