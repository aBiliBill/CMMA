# CMMA

Code for **Complementary Local–Global Matching With Multi-Scale Context Alignment for Cross-View Geo-Localization**.

Complementary local–global Matching with Multi-scale context Alignment (CMMA) combines a shared ConvNeXt-Base encoder, training-time multi-scale context alignment (MCFA), complementary local matching (CLM), and bidirectional multi-positive ranking. SSC is disabled.

This release contains source code, dependencies and usage instructions. Datasets, pretrained or trained weights, experiment logs, cached features and manuscripts are not included.

## Installation

Use Python **3.10** in a separate environment. Install PyTorch 2.7.1 and torchvision 0.22.1 for your platform using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/), then install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
python example.py
```

`example.py` runs a small CPU forward/backward check with synthetic images, followed by descriptor and retrieval-metric checks. It does not download data or weights. Full training uses 384 x 384 images and is intended for a CUDA GPU; the synthetic check does not establish benchmark accuracy.

## Code layout

| File | Purpose |
| --- | --- |
| `backbone.py` | Shared ConvNeXt encoder |
| `context.py` | Three dilated MCFA branches |
| `model.py` | Descriptors and the joint training objective |
| `retrieval.py` | Directed local matching and multi-positive ranking |
| `data.py`, `sampling.py` | Image augmentation, location batches and auxiliary drone views |
| `train.py` | One-epoch training entry point |
| `evaluate.py`, `metrics.py` | Directional retrieval and trapezoidal AP |
| `evaluate_transfer.py` | Both transfer directions; four-source-height mean for SUES to University |
| `checkpoints.py` | Local pretrained initialization and checkpoint save/load |
| `example.py` | Small synthetic method check |

## Data preparation

Obtain [University-1652](https://github.com/layumi/University1652-Baseline) or [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) from the dataset providers. Respect their access and usage conditions.

Pass directories with one subdirectory per numeric location ID:

```text
DATASET/
  train/
    satellite/0001/image.jpg
    drone/0001/view_01.jpg
    drone/0001/view_02.jpg
    ...
  test/
    query_drone/...
    gallery_satellite/...
    query_satellite/...
    gallery_drone/...
```

For University-1652 these are the official `train` and `test` directories. For SUES-200 use `Training/HEIGHT` and `Testing/HEIGHT`, where HEIGHT is 150, 200, 250 or 300. The diagram describes directory roles; do not merge training and test locations or move test images into training.

Training requires at least 24 shared locations, one satellite image and at least two distinct drone images per location. Keep only images in the location directories. Each main batch contains 24 distinct locations; ranking adds a second drone view from each of those locations.

On Windows, keep image datasets under an ASCII-only path (for example `E:/datasets/SUES-200`). The retained OpenCV image loader may not open paths containing non-ASCII characters.

## ImageNet initialization

Download **ConvNeXt-Base, ImageNet-22K pretraining followed by ImageNet-1K fine-tuning, 224 resolution** from the [official ConvNeXt model release](https://github.com/facebookresearch/ConvNeXt). The checkpoint is named `convnext_base_22k_1k_224.pth` and is available at:

<https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth>

Expected SHA256:

```text
8afbd1fa254629e6db80df9822908a83c741bfe98778f58851a15c1639a80453
```

Store the file locally under `weights/` or another directory outside the repository. Training checks this hash and loads the encoder strictly. The checkpoint is not bundled, and the scripts do not download it automatically. The initialization file is trained at 224 resolution; this method fine-tunes and evaluates at 384 resolution.

## Training

Example for SUES-200 at 250 m:

```bash
python train.py --satellite DATA/SUES-200/Training/250/satellite --drone DATA/SUES-200/Training/250/drone --pretrained weights/convnext_base_22k_1k_224.pth --seed 1 --output outputs/sues250/model.pth
```

For the University-1652 within-dataset experiments, use seed 4:

```bash
python train.py --satellite DATA/University-1652/train/satellite --drone DATA/University-1652/train/drone --pretrained weights/convnext_base_22k_1k_224.pth --seed 4 --output outputs/university_seed4/model.pth
```

The entry point uses the source supplement's one-epoch recipe: AdamW, learning rate 0.001, weight decay 0.01, batch size 24, cosine decay with 10% warmup, and CUDA mixed precision when available. The ranking temperature is 0.1; its coefficient increases to 0.1 over the first 10 updates. No validation or test data are used to select a checkpoint. Only the final model is saved; an existing output file is protected from overwriting.

## Evaluation

Drone to satellite:

```bash
python evaluate.py --checkpoint outputs/sues250/model.pth --query DATA/SUES-200/Testing/250/query_drone --gallery DATA/SUES-200/Testing/250/gallery_satellite
```

Satellite to drone:

```bash
python evaluate.py --checkpoint outputs/sues250/model.pth --query DATA/SUES-200/Testing/250/query_satellite --gallery DATA/SUES-200/Testing/250/gallery_drone
```

The commands print `R@1` and `AP` as percentages. `AP` is the mean of the original **trapezoidal** per-query AP, not the alternate standard precision-at-positive-ranks definition. Each query is scored against the full physical gallery, sorted, then filtered to locations present in the query set. This yields the effective 80-location protocol for SUES-200 and 701-location protocol for University-1652. Results from protocols retaining other-location distractors are not interchangeable.

Satellite-to-drone retrieval recomputes the directed local matching score. Add `--device cpu` for CPU evaluation if needed. The single-direction `evaluate.py` defaults to FP32 feature extraction; `--amp` enables CUDA AMP extraction, and `--batch-size` controls the image batch size. Descriptors are converted to FP32 before scoring.

## Cross-dataset evaluation

Use only the final checkpoint trained on the source dataset. Do not train or fine-tune on the target dataset, or select a source checkpoint or source height using target test performance. The commands below use checkpoints produced by this release's `train.py`. Checkpoints from the earlier public code release remain supported; internal historical experiment checkpoints use a different structure and cannot be loaded directly.

### University-1652 to SUES-200

The University-1652-to-SUES-200 transfer experiment uses a separate source model trained with seed 1. Evaluate the same source checkpoint on all four SUES-200 test heights:

```bash
python train.py --satellite DATA/University-1652/train/satellite --drone DATA/University-1652/train/drone --pretrained weights/convnext_base_22k_1k_224.pth --seed 1 --output outputs/university_seed1/model.pth

python evaluate_transfer.py university-to-sues --checkpoint outputs/university_seed1/model.pth --sues-test DATA/SUES-200/Testing --device cuda --precision amp
```

Both drone-to-satellite (`D2S`) and satellite-to-drone (`S2D`) retrieval are evaluated separately at 150, 200, 250 and 300 m. The output retains a separate result for each target height, as in the height-specific comparison tables.

### SUES-200 to University-1652

Train four independent source models, one per SUES-200 height, with the same fixed seed and training recipe. The following is a Bash example; each training command can also be run individually:

```bash
for height in 150 200 250 300; do
  python train.py --satellite DATA/SUES-200/Training/${height}/satellite --drone DATA/SUES-200/Training/${height}/drone --pretrained weights/convnext_base_22k_1k_224.pth --seed 1 --output outputs/sues${height}/model.pth
done
```

Evaluate all four final models on the same University-1652 test directories:

```bash
python evaluate_transfer.py sues-to-university --checkpoint-150 outputs/sues150/model.pth --checkpoint-200 outputs/sues200/model.pth --checkpoint-250 outputs/sues250/model.pth --checkpoint-300 outputs/sues300/model.pth --university-test DATA/University-1652/test --device cuda --precision amp
```

For every source model, the script independently computes `D2S` and `S2D` R@1 and AP. It then takes an equal arithmetic mean across the four source heights for each of the four metrics, using the unrounded values. These are averages of separately evaluated models, not an ensemble, a pooled training run, or the best of four models. All four checkpoint arguments are required; supplying the same resolved path more than once is rejected. The caller must ensure each file was trained on its stated source height.

The JSON printed to the terminal includes `per_height` and, for SUES-to-University, `mean`. Metric values are percentages; `AP` retains the original trapezoidal definition and the same query-location gallery filtering described above. Both retrieval directions are recomputed, rather than transposing directional local scores. No experimental scores are hard-coded or bundled. The script does not write weights, logs, feature caches or result files.

Both transfer commands default to AMP feature extraction on CUDA, or FP32 on CPU; scoring remains FP32. Use `--device cpu --precision fp32` for a CPU check. These precisions need not yield numerically identical rankings. `--batch-size 16` is the default. The standalone evaluator keeps extracted descriptors in RAM, so large galleries require sufficient host memory even when GPU batches are small.

## Release scope

This is the existing standalone source supplement with working local checkpoint and command-line interfaces. Core model, loss, sampling and metric code is retained. Its training/data-loader and feature-extraction interfaces are simplified relative to the experiment runners; it is not a byte-identical export of those runners. Checkpoint keys use this release's `encoder` / `context` structure. New checkpoints use the `cmma-code-v1` format identifier, and the earlier public format remains loadable. Internal historical experiment checkpoint dictionaries are not directly interchangeable. The auxiliary random-seed namespace is preserved to keep the sampled training views unchanged by the method rename.

The packaged entry point implements the one-epoch recipe. The complete ablation suite and multi-seed experiment orchestration are outside this source release. The transfer wrapper adds experiment orchestration and metric averaging, not historical checkpoint conversion. Optional AMP extraction does not make this standalone implementation numerically identical to the experiment runners: local descriptor processing and score accumulation still follow the standalone implementation. Benchmark numbers are not embedded in this code release. Training and retrieval can vary with random seed, hardware and numerical implementation; a fresh full benchmark reproduction has not been performed for this packaging revision.

Locally generated outputs may contain weights and measurements. `.gitignore` excludes those files from normal Git staging; this does not filter files manually uploaded through a browser. Upload only the prepared source files.

## Acknowledgements and licenses

The encoder is adapted from [Meta ConvNeXt](https://github.com/facebookresearch/ConvNeXt), under its MIT license in `LICENSE-ConvNeXt.txt`. The loader and transformations build on [Sample4Geo](https://github.com/Skyy93/Sample4Geo) and the supplied project code. The supplied Apache-2.0 license is retained in `LICENSE-project.txt`. See `THIRD_PARTY_NOTICES.txt` for component attribution. Existing license texts and copyright notices are preserved.
