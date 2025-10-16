#[macro_use]
extern crate criterion;
extern crate tract_data;

use criterion::{black_box, Criterion};
use tract_data::internal::*;

fn symbol_clone(c: &mut Criterion) {
    let scope = SymbolScope::default();
    let s = scope.sym("S");
    c.bench_function("symbol_clone_1k", |b| {
        b.iter(|| {
            let mut x = s;
            for _ in 0..1000 {
                // prevent the compiler from optimizing the loop away
                x = black_box(x.clone());
            }
            black_box(x)
        })
    });
}

criterion_group!(benches, symbol_clone);
criterion_main!(benches);
