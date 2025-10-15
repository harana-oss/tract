use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

fn bench_softmax_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_inplace_rows");
    // Test a few representative shapes: small, medium, large number of columns
    let shapes = [(128usize, 8usize), (128, 32), (64, 128), (32, 512), (16, 2048)];

    for (rows, cols) in shapes {
        let mut rng = StdRng::seed_from_u64(42);
        // Pristine source data, and a working buffer reused across iters
        let src: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let mut buf = src.clone();
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", rows, cols)),
            &cols,
            |b, &cols| {
                b.iter_batched(
                    || {
                        // Fresh working buffer each iter
                        let mut working = src.clone();
                        std::hint::black_box(working)
                    },
                    |mut d| {
                        tract_onnx_opl::ml::softmax::softmax_inplace_rows(&mut d, rows, cols);
                        std::hint::black_box(())
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Also benchmark a strided/gather input pattern using a padded source buffer
        let pad = 7usize; // arbitrary small padding to create a non-contiguous stride
        let stride = cols + pad;
        let mut src_gather: Vec<f32> = vec![0.0; rows * stride];
        for r in 0..rows {
            let start = r * stride;
            let slice = &mut src_gather[start..start + cols];
            slice.copy_from_slice(&src[r * cols..(r + 1) * cols]);
        }
        let mut out = vec![0f32; rows * cols];
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("gather-{}x{}(s0={},s1=1)", rows, cols, stride)),
            &cols,
            |b, &_cols| {
                b.iter(|| unsafe {
                    tract_onnx_opl::ml::softmax::softmax_rows_gather_into_tl(
                        src_gather.as_ptr(),
                        rows,
                        cols,
                        stride as isize,
                        1,
                        &mut out,
                    );
                    std::hint::black_box(&out);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_softmax_rows);
criterion_main!(benches);
