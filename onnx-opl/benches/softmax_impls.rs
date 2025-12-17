use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
// removed: standardized on single softmax implementation; this bench is obsolete.
fn bench_softmax_impls(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_impls");
    let shapes = [(128usize, 16usize), (64, 128), (32, 512), (16, 2048)];

    for (rows, cols) in shapes {
        let mut rng = StdRng::seed_from_u64(2024);
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::new("baseline", format!("{}x{}", rows, cols)), &cols, |b, &_cols| {
            b.iter_batched(
                || src.clone(),
                |mut v| {
                    tract_onnx_opl::ml::softmax::softmax_inplace_rows_baseline(&mut v, rows, cols);
                    std::hint::black_box(&v);
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("optimized", format!("{}x{}", rows, cols)), &cols, |b, &_cols| {
            b.iter_batched(
                || src.clone(),
                |mut v| {
                    tract_onnx_opl::ml::softmax::softmax_inplace_rows_optimized(&mut v, rows, cols);
                    std::hint::black_box(&v);
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("scalar", format!("{}x{}", rows, cols)), &cols, |b, &_cols| {
            b.iter_batched(
                || src.clone(),
                |mut v| {
                    tract_onnx_opl::ml::softmax::softmax_inplace_rows_scalar(&mut v, rows, cols);
                    std::hint::black_box(&v);
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_softmax_impls);
criterion_main!(benches);
