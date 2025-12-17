use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

fn bench_linear_regressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_linear_regressor");

    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH to an existing .onnx file containing ai.onnx.ml.LinearRegressor");
    let onnx_path = PathBuf::from(&model_path);
    assert!(onnx_path.exists(), "Model path does not exist: {}", model_path);

    // Load model once for shape inference
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();
    let n = model.sym("N");
    let model =
        model.with_input_fact(0, f32::fact(dims!(n, 5)).into()).unwrap().into_optimized().unwrap();

    let input_fact = model.input_fact(0).unwrap().clone();
    let shape: TVec<usize> = input_fact
        .shape
        .as_concrete()
        .map(|s| s.iter().copied().collect())
        .unwrap_or_else(|| tvec![1, 5]);
    let num_features = shape[1];

    // Pre-generate random input tensors
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let input_tensors: Arc<Vec<Tensor>> = Arc::new(
        (0..281_539)
            .map(|_| {
                let sample: Vec<f32> =
                    (0..num_features).map(|_| rng.gen_range(-30.0f32..30.0f32)).collect();
                Tensor::from_shape(&shape, &sample).unwrap()
            })
            .collect(),
    );

    group.bench_function(
        BenchmarkId::new("load_opt_run_parallel", onnx_path.display().to_string()),
        |b| {
            let runnable = Arc::new(model.clone().into_runnable().unwrap());
            let tensors = Arc::clone(&input_tensors);

            b.iter_custom(|_| {
                let start = Instant::now();

                (0..281_539usize).into_par_iter().for_each(|i| {
                    let runnable = Arc::clone(&runnable);
                    let input_val = tensors[i].clone().into_tvalue();
                    let _ = runnable.run(tvec!(input_val)).unwrap();
                });

                start.elapsed()
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_linear_regressor);
criterion_main!(benches);
