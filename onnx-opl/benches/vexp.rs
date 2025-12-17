use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
// removed: standardized on single exp/softmax implementation; this bench is obsolete.
fn bench_vexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("vexp_f32_variants");
    let sizes = [1usize << 12, 1 << 16, 1 << 20]; // 4K, 64K, 1M elements

    for &n in &sizes {
        let mut rng = StdRng::seed_from_u64(123);
        let mut src: Vec<f32> = (0..n).map(|_| rng.gen_range(-30.0..0.0)).collect();
        group.throughput(Throughput::Elements(n as u64));

        // softmax_fast
        group.bench_with_input(BenchmarkId::new("softmax_fast", n), &n, |b, _| {
            b.iter(|| {
                let mut v = src.clone();
                tract_onnx_opl::ml::softmax::vexp_inplace_softmax_fast(&mut v);
                std::hint::black_box(&v);
            })
        });

        // poly4
        group.bench_with_input(BenchmarkId::new("poly4", n), &n, |b, _| {
            b.iter(|| {
                let mut v = src.clone();
                tract_onnx_opl::ml::softmax::vexp_inplace_poly4(&mut v);
                std::hint::black_box(&v);
            })
        });

        // minimax6
        group.bench_with_input(BenchmarkId::new("minimax6", n), &n, |b, _| {
            b.iter(|| {
                let mut v = src.clone();
                tract_onnx_opl::ml::softmax::vexp_inplace_minimax6(&mut v);
                std::hint::black_box(&v);
            })
        });

        // minimax7
        group.bench_with_input(BenchmarkId::new("minimax7", n), &n, |b, _| {
            b.iter(|| {
                let mut v = src.clone();
                tract_onnx_opl::ml::softmax::vexp_inplace_minimax7(&mut v);
                std::hint::black_box(&v);
            })
        });

        // scalar
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                let mut v = src.clone();
                tract_onnx_opl::ml::softmax::vexp_inplace_scalar(&mut v);
                std::hint::black_box(&v);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vexp);
criterion_main!(benches);
