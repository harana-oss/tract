use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

#[cfg(target_arch = "aarch64")]
fn bench_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax");

    // Per-row argmax benchmark with varying lengths
    for cols in [8usize, 32, 128, 512, 2048] {
        let mut rng = StdRng::seed_from_u64(123);
        let row: Vec<f32> = (0..cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
        group.throughput(Throughput::Elements(cols as u64));
        group.bench_with_input(BenchmarkId::new("argmax_row", cols), &cols, |b, &_cols| {
            b.iter(|| {
                let idx = tract_onnx_opl::ml::argmax::argmax_row(std::hint::black_box(&row));
                std::hint::black_box(idx)
            })
        });
    }

    // Multi-row argmax benchmark
    for (rows, cols) in [(128usize, 32usize), (64, 128), (32, 512), (16, 2048)] {
        let mut rng = StdRng::seed_from_u64(321);
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("argmax_rows", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter(|| {
                    let out = tract_onnx_opl::ml::argmax::argmax_rows(
                        std::hint::black_box(&data),
                        rows,
                        cols,
                    );
                    std::hint::black_box(out)
                })
            },
        );

        // Gather/strided variant
        let pad = 5usize;
        let stride = cols + pad;
        let mut data_g = vec![0f32; rows * stride];
        for r in 0..rows {
            data_g[r * stride..r * stride + cols].copy_from_slice(&data[r * cols..(r + 1) * cols]);
        }
        group.bench_with_input(
            BenchmarkId::new(
                "argmax_rows_gather",
                format!("{}x{}(s0={},s1=1)", rows, cols, stride),
            ),
            &(rows, cols),
            |b, &(rows, _cols)| {
                b.iter(|| unsafe {
                    let out = tract_onnx_opl::ml::argmax::argmax_rows_gather(
                        data_g.as_ptr(),
                        rows,
                        cols,
                        stride as isize,
                        1,
                    );
                    std::hint::black_box(out)
                })
            },
        );
    }

    group.finish();
}
#[cfg(target_arch = "aarch64")]
criterion_group!(benches, bench_argmax);
#[cfg(target_arch = "aarch64")]
criterion_main!(benches);

// Fallback to allow compiling this bench on non-aarch64 targets
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("argmax benches require aarch64 target");
}
