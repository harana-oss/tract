use tract_nnef::internal::*;

pub mod argmax;
pub mod category_mapper;
pub mod linear_classifier;
pub mod linear_regressor;
pub mod math;
pub mod matmul;
pub mod normalizer;
pub mod smallvec;
pub mod softmax;
pub mod tree;
pub mod tree_ensemble_classifier;

pub use category_mapper::{DirectLookup, ReverseLookup};

pub fn register(registry: &mut Registry) {
    category_mapper::register(registry);
    linear_classifier::register(registry);
    linear_regressor::register(registry);
    normalizer::register(registry);
    tree_ensemble_classifier::register(registry);
}
