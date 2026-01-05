use actix::prelude::*;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::future::join_all;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use par_iter::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tract_hir::internal::*;
use tract_onnx::tract_core::dims;

#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

// Simple Actix actor that owns a Runnable and references input tensors; it executes one
// inference per message for the provided input index.
struct InferenceActor {
    runnable: tract_hir::internal::TypedRunnableModel<tract_hir::internal::TypedModel>,
    inputs: Arc<Vec<Tensor>>,
}

impl Actor for InferenceActor {
    type Context = Context<Self>;
}

// Message carrying an input index (Send), the actor fetches the Tensor from its Arc store
#[derive(Clone, Copy)]
struct InferIdx(pub usize);

impl Message for InferIdx {
    type Result = ();
}

impl Handler<InferIdx> for InferenceActor {
    type Result = ();

    fn handle(&mut self, msg: InferIdx, _ctx: &mut Self::Context) -> Self::Result {
        // Fetch tensor by index, convert to TValue, and run once; ignore output
        let input_val = self.inputs[msg.0].clone().into_tvalue();
        let _ = self.runnable.run(tvec!(input_val));
    }
}

fn bench_linear_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_linear_classifier");
    group.sample_size(10);

    let model_path = std::env::var("MODEL_PATH")
        .expect("Set MODEL_PATH to an existing .onnx file containing ai.onnx.ml.LinearClassifier");
    let onnx_path = PathBuf::from(&model_path);
    assert!(onnx_path.exists(), "Model path does not exist: {}", model_path);

    // Load model once for shape inference
    let model = tract_onnx::onnx().model_for_path(&onnx_path).unwrap();
    let n = model.sym("N");

    // Configure input and output dimensions
    let input_dim: usize = std::env::var("INPUT")
        .expect("Set INPUT to the input feature dimension (usize)")
        .parse()
        .expect("INPUT must be a positive integer");
    let output_dim: usize = std::env::var("OUTPUT")
        .expect("Set OUTPUT to the number of classes (usize)")
        .parse()
        .expect("OUTPUT must be a positive integer");

    // Configure parallel iterator chunking via env var (default: 1000)
    // Set WITH_MIN_LEN to control Rayon with_min_len to tune parallelism granularity.
    let min_len: usize =
        std::env::var("WITH_MIN_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);

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

    // Pre-generate random input tensors
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

    group.bench_function(
        BenchmarkId::new("load_opt_run_parallel", onnx_path.display().to_string()),
        |b| {
            let tensors = Arc::clone(&input_tensors);

            b.iter_custom(|_| {
                let start = Instant::now();

                (0..15_000).for_each(|i| {
                    let input_val = tensors[i].clone().into_tvalue();
                    let runnable = model.clone().into_runnable().unwrap();

                    for _ in 0..2800 {
                        let _ = runnable.run(tvec!(input_val.clone())).unwrap();
                    }
                });

                start.elapsed()
            });
        },
    );

    // New Actix variant: 10 actors, each called 28 times serially per input (configurable via env)
    // group.bench_function(
    //     BenchmarkId::new("load_opt_run_actix_actors", onnx_path.display().to_string()),
    //     |b| {
    //         let tensors = Arc::clone(&input_tensors);

    //         // Allow override via env for experiments
    //         let actor_count: usize =
    //             std::env::var("ACTOR_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    //         let calls_per_actor: usize =
    //             std::env::var("CALLS_PER_ACTOR").ok().and_then(|s| s.parse().ok()).unwrap_or(28);

    //         b.iter_custom(|_| {
    //             let start = Instant::now();

    //             // Run a fresh Actix system per measurement to keep lifecycle simple
    //             actix_rt::System::new().block_on(async {
    //                 // Create one Runnable per actor
    //                 let mut actors: Vec<Addr<InferenceActor>> = Vec::with_capacity(actor_count);
    //                 for _ in 0..actor_count {
    //                     let runnable = model.clone().into_runnable().unwrap();
    //                     actors.push(
    //                         InferenceActor { runnable, inputs: Arc::clone(&tensors) }.start(),
    //                     );
    //                 }

    //                 // For each input tensor, call each actor `calls_per_actor` times serially
    //                 for i in 0..15_000usize {
    //                     let futs = actors.iter().cloned().map(|addr| {
    //                         async move {
    //                             for _ in 0..calls_per_actor {
    //                                 // Ignore mailbox errors for benchmark simplicity
    //                                 let _ = addr.send(InferIdx(i)).await;
    //                             }
    //                         }
    //                     });
    //                     join_all(futs).await;
    //                 }
    //             });

    //             start.elapsed()
    //         });
    //     },
    // );

    group.finish();
}

criterion_group!(benches, bench_linear_classifier);
criterion_main!(benches);
