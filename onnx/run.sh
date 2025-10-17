#!/bin/bash
MODEL_PATH="/Users/nadenf/Developer/dp-tract/onnx/test_cases/linear_classifier/model.onnx" \
INPUT=12 \
OUTPUT=14 \
SAMPLES=15000 \
STEPS=28000 \
../target/debug/linear_classifier
