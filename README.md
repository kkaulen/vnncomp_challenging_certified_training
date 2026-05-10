# VNN-COMP Challenging Certified Training

This benchmark contains verification instances for six certified-training models:

- `cifar10_eps2_cnn7`
- `cifar10_eps2_wide_cnn7`
- `cifar10_eps8_cnn7`
- `cifar10_eps8_wide_cnn7`
- `tinyimagenet_eps1_cnn7`
- `tinyimagenet_eps1_wide_cnn7`

The benchmark is self-contained for VNN-COMP use. The committed `onnx/`,
`onnx_models.zip`, `verification_results/`, `vnnlib/`, `instances.csv`, and
`metadata/sampled_instances.json` files are enough to run the benchmark on
another machine.

## Benchmark Generation

Generate a seed-specific benchmark from bundled verification results and test
sets with:

```bash
python generate_properties.py 42
```

This default mode does not need the original training checkpoints or the
original abCROWN workspace. It does the following:

- ensures the six ONNX models exist, extracting `onnx_models.zip` if needed
- if the zip is absent, downloads it from `--onnx-zip-url`,
  `VNNCOMP_ONNX_ZIP_URL`, or the built-in Sciebo share
- downloads CIFAR-10 and TinyImageNet under `data/` when missing
- tries CTRAIN dataset loaders first, then falls back to local
  torchvision/urllib download helpers
- samples new image indices from the bundled `verification_results/*.json`
  files using the requested seed
- regenerates `vnnlib/`, `instances.csv`, and `metadata/sampled_instances.json`

For an external ONNX archive:

```bash
python generate_properties.py 42 --onnx-zip-url https://example.org/onnx_models.zip
```

The built-in ONNX archive source is the password-protected Sciebo share
`https://rwth-aachen.sciebo.de/s/zr2GXGNWwjyWrBX`. The default password is
included in the generator, and can be overridden with `VNNCOMP_SCIEBO_PASSWORD`.

To use an existing dataset directory:

```bash
python generate_properties.py 42 --data-root /path/to/data
```

The generator writes:

- `instances.csv`
- `vnnlib/<model_key>/*.vnnlib`
- `metadata/sampled_instances.json`

The sampling layout is:

- 60 total instances
- 10 instances per model
- per model: 2 from `[0,10]`, 2 from `[10,100]`, 3 from `[100,1000]`, and 3
  timeout instances
- total CSV timeout: 21,600 seconds

Per-instance CSV timeouts are:

- `[0,10]`: 30 seconds
- `[10,100]`: 120 seconds
- `[100,1000]`: 550 seconds
- `timeout`: 550 seconds

## Source Rebuild

Development-only full regeneration is available with:

```bash
python generate_properties.py 42 --rebuild-from-sources
```

This mode recreates `onnx/`, `vnnlib/`, `instances.csv`, and metadata from the
original local training workspace. It expects the checkpoint paths under
`../results`. Verification results still come from the benchmark-local
`verification_results/` files.

The source rebuild intentionally keeps local model definitions in this script so
that checkpoint-to-ONNX export is independent of broader experiment side
effects. The source checkpoints are CTRAIN `ModelWrapper.state_dict()` files,
which delegate to auto_LiRPA `BoundedModule.state_dict()`. Their keys are graph
node names such as `/1.param` and `/5.buffer`; the generator maps these tensors
to the local PyTorch model by ordered, shape-checked assignment. BatchNorm
weights, biases, running means, and running variances are all loaded. The only
PyTorch BatchNorm state entries not present in the auto_LiRPA checkpoints are
`num_batches_tracked` counters.

The ONNX export path mirrors CTRAIN's helper: opset 18, dynamic batch axis,
evaluation mode, `dynamo=False`, and post-export removal of BatchNormalization
or Dropout `training_mode` attributes.

CTRAIN can be installed alongside this benchmark for development workflows, but
it is not required to run the committed benchmark artifacts.
