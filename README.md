# VNN-COMP Challenging Certified Training

This benchmark targets complete verification of recent state-of-the-art
certified-training models trained with [CTRAIN](https://github.com/ADA-research/CTRAIN). It contains six MTL-IBP-based
models ([De Palma et al.](https://arxiv.org/abs/2305.13991)) and seed-specific local-robustness properties for CIFAR-10 and
TinyImageNet.

## Motivation
Using [CTRAIN](https://github.com/ADA-research/CTRAIN), we obtained unusually strong certifiably trained models based on
MTL-IBP ([De Palma et al.](https://arxiv.org/abs/2305.13991)). These models improve the state of the art in certified training, but
they also expose a verification bottleneck: better certified-training
performance often makes complete verification substantially harder.

| Setting | Model | Standard acc. | Certified acc. |
| --- | --- | ---: | ---: |
| CIFAR-10, epsilon `2/255` | `cnn7` | `83.54%` | `66.04%` |
| CIFAR-10, epsilon `8/255` | `cnn7` | `57.19%` | `35.49%` |
| TinyImageNet, epsilon `1/255` | `cnn7_tinyimagenet` | `41.59%` | `27.81%` |

We additionally include wider variants of these models. These models may be
state of the art in terms of certified-training performance, but they suffer
severely from timeouts under complete verification.

| Setting | Wide model | Standard acc. | Certified acc. |
| --- | --- | ---: | ---: |
| CIFAR-10, epsilon `2/255` | `wide_cnn7` | `85.37%` | `60.28%` |
| CIFAR-10, epsilon `8/255` | `wide_cnn7` | `57.74%` | `34.60%` |
| TinyImageNet, epsilon `1/255` | `wide_cnn7_tinyimagenet` | `41.89%` | `28.35%` |

The bundled results were obtained with [$\alpha\beta$-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) using its standard configuration on a NVIDIA H100 GPU and a per-instance cutoff time of $1\,000$ seconds. Each of the six benchmark models was
evaluated on `10,000` test-set properties. Across all `60,000` properties,
`2,994` timed out. The hardest individual setting is `cifar10_eps2_wide_cnn7`,
with `1,433/10,000` timeouts.

| Model | Evaluated properties | Timeouts | Timeout rate |
| --- | ---: | ---: | ---: |
| `cifar10_eps2_cnn7` | `10,000` | `658` | `6.58%` |
| `cifar10_eps2_wide_cnn7` | `10,000` | `1,433` | `14.33%` |
| `cifar10_eps8_cnn7` | `10,000` | `134` | `1.34%` |
| `cifar10_eps8_wide_cnn7` | `10,000` | `132` | `1.32%` |
| `tinyimagenet_eps1_cnn7` | `10,000` | `236` | `2.36%` |
| `tinyimagenet_eps1_wide_cnn7` | `10,000` | `401` | `4.01%` |
| Total | `60,000` | `2,994` | `4.99%` |

In other words, progress in certified training now creates verification
workloads that current state-of-the-art verifiers often cannot finish within practical
budgets.

**This benchmark is designed to measure progress on complete verification for
challenging but highly relevant certifiably trained image classifiers.**

## Methodology

We start from three strong CTRAIN certified-training models and include their
matching wide-model variants:

- CIFAR-10, epsilon `2/255`, `cnn7`.
- CIFAR-10, epsilon `2/255`, `wide_cnn7`.
- CIFAR-10, epsilon `8/255`, `cnn7`.
- CIFAR-10, epsilon `8/255`, `wide_cnn7`.
- TinyImageNet, epsilon `1/255`, `cnn7_tinyimagenet`.
- TinyImageNet, epsilon `1/255`, `wide_cnn7_tinyimagenet`.

Each model was evaluated on `10,000` test-set properties with abCROWN using its
standard complete-verification configuration. We use the resulting
`verification_results/*.json` files to stratify properties by observed abCROWN
runtime:

- `[0,10]` seconds
- `[10,100]` seconds
- `[100,1000]` seconds
- timeout

For each model, the generator samples `10` properties:

- 2 from `[0,10]`
- 2 from `[10,100]`
- 3 from `[100,1000]`
- 3 from timeout

This yields `60` total benchmark instances. If a bin has fewer properties than
requested for a model, sampling is performed with replacement and recorded in
the metadata. The generated CSV timeout budget is capped at 6 hours exactly:

- `[0,10]`: `30` seconds
- `[10,100]`: `120` seconds
- `[100,1000]`: `550` seconds
- timeout: `550` seconds

Total timeout budget:
`12 * 30 + 12 * 120 + 18 * 550 + 18 * 550 = 21,600` seconds.

The generated VNN-LIB files encode standard local robustness. Inputs are
normalized test images, perturbations are channelwise normalized by the dataset
standard deviation, and bounds are clamped to normalized image-space `[0,1]`
limits. The output constraints encode the adversarial disjunction, so `unsat`
means the model is robust for the property.

## Benchmark Generation

The repository ships only the benchmark generator and the abCROWN verification
results used for sampling. ONNX models, datasets, VNN-LIB files, metadata, and
`instances.csv` are generated locally.

Generate a seed-specific benchmark with:

```bash
python generate_properties.py 42
```

The generator:

- downloads the six ONNX models from `--onnx-zip-url`,
  `VNNCOMP_ONNX_ZIP_URL`, or the built-in Sciebo share
- downloads CIFAR-10 and TinyImageNet test data under `data/` when missing
- samples new image indices from `verification_results/*.json`
- writes `onnx/`, `vnnlib/`, `instances.csv`, and
  `metadata/sampled_instances.json`

For an external ONNX archive:

```bash
python generate_properties.py 42 --onnx-zip-url https://example.org/onnx_models.zip
```

The built-in ONNX archive source is the password-protected Sciebo share
`https://rwth-aachen.sciebo.de/s/zr2GXGNWwjyWrBX`. The default password is
included in the generator and can be overridden with `VNNCOMP_SCIEBO_PASSWORD`.

To use an existing dataset directory:

```bash
python generate_properties.py 42 --data-root /path/to/data
```
