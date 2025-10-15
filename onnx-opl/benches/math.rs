use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use std::hint::black_box;

fn bench_dot_and_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_neon");

    // Dot product sizes
    for &len in &[16usize, 64, 256, 1024, 4096] {
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<f32> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("dot_neon", len), &len, |bch, &_len| {
            bch.iter(|| unsafe {
                let s = tract_onnx_opl::ml::math::dot_neon(black_box(&a), black_box(&b));
                black_box(s)
            });
        });
    }

    // Matmul row-wise sizes (n rows, c cols, e weight vectors)
    for &(n, c, e) in &[(32usize, 64usize, 8usize), (32, 128, 32), (16, 512, 16)] {
        let mut rng = StdRng::seed_from_u64(2);
        let x: Vec<f32> = (0..n * c).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w: Vec<f32> = (0..e * c).map(|_| rng.gen_range(-1.0..1.0)).collect();
        group.throughput(Throughput::Elements((n * c) as u64));
        group.bench_with_input(
            BenchmarkId::new("matmul_rows_contig", format!("{}x{}x{}", n, c, e)),
            &(n, c, e),
            |bch, &(n, c, e)| {
                bch.iter_batched(
                    || vec![0f32; n * e],
                    |mut out| unsafe {
                        tract_onnx_opl::ml::math::matmul_rows_neon_contig(
                            &x, n, c, &w, e, &mut out,
                        );
                        black_box(())
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Fake strided/gather path: use s0=c, s1=1 but read through pointer
        group.bench_with_input(
            BenchmarkId::new("matmul_rows_gather", format!("{}x{}x{}", n, c, e)),
            &(n, c, e),
            |bch, &(n, c, e)| {
                bch.iter_batched(
                    || (vec![0f32; n * e], vec![0f32; c]),
                    |(mut out, mut rowbuf)| unsafe {
                        let base = x.as_ptr();
                        tract_onnx_opl::ml::math::matmul_rows_neon_gather(
                            base,
                            n,
                            c,
                            c as isize,
                            1,
                            &w,
                            e,
                            &mut out,
                            &mut rowbuf,
                        );
                        black_box(())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dot_and_matmul);
criterion_main!(benches);
