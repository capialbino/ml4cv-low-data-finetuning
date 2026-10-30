# Low-data fine-tuning of ViT-B/16: linear probe, partial FT, full FT, LoRA

ML4CV project work (University of Bologna, Prof. S. Salti). All code, tables, figures, findings and the discussion are in the notebook [`low_data_finetuning.ipynb`](low_data_finetuning.ipynb), which is committed with its outputs.

## Question

How does the best way to adapt a pre-trained ViT-B/16 change with the amount of labelled data, and does the answer depend on the pre-training (CLIP or supervised ImageNet-1k)?

Hypotheses, written before any experiment was run:

- **H1.** With 1-5 examples per class, linear probe and LoRA are at least as good as full fine-tuning.
- **H2.** With more data full fine-tuning catches up and eventually wins: there is a crossover point.
- **H3.** CLIP features are better than supervised ImageNet features in the low-shot regime.

## Setup

| | |
|---|---|
| Backbones (`timm`) | CLIP ViT-B/16 image tower (`vit_base_patch16_clip_quickgelu_224.openai`); supervised ViT-B/16 trained on ImageNet-1k only (`vit_base_patch16_224.augreg_in1k`). New linear head initialised at zero. |
| Methods | linear probe (7.7k trainable parameters for 10 classes); partial FT, last 4 blocks (28.4M); LoRA rank 8 on `qkv` and `proj` (0.45M); full FT (85.8M) |
| Datasets | Flowers102 (102 classes), Oxford-IIIT Pets (37), EuroSAT RGB (10, satellite images) |
| Training | AdamW, bf16 autocast, warm-up + cosine decay, 10 epochs with at least 100 and at most 1,500 steps |
| Learning rate | fixed in Experiment 1; chosen on a validation set (seed 0, grid of 4-6 values per method) in Experiments 3 and 4 |
| Statistics | 3 seeds; the same seed gives the same k-shot subset to every method and backbone, so differences are paired by seed. A difference is called *clear* only if it has the same sign on all seeds, is larger than its std and is larger than about 0.01. |

## Experiments

| | content |
|---|---|
| 1. Main study | Flowers102 and Pets, 1, 5 and all images per class, fixed learning rates (exploratory) |
| 2. Replication | CLIP-LoRA (Zanella & Ben Ayed, 2024) re-implemented from the paper and its official code, as a check of the pipeline |
| 3. EuroSAT scaling | 1 to 1,500 images per class, learning rate tuned per method and size, both backbones |
| 4. Tuned Flowers102 / Pets | the protocol of Experiment 3 on the datasets of Experiment 1 (up to 10 and 20 images per class) |
| 6.1-6.4 Ablations and checks | LoRA rank, number of unfrozen blocks, linear-probe budget; training budget; sensitivity to the learning-rate choice; extended learning-rate grid; logistic-regression (L-BFGS) linear probe as in the CLIP paper |

## Main results

**EuroSAT, test accuracy at the validation-selected learning rate (mean of 3 seeds)**

| images/class | 1 | 5 | 20 | 100 | 500 | 1,500 |
|---|---|---|---|---|---|---|
| CLIP, full FT | 0.634 | 0.836 | 0.924 | 0.959 | 0.977 | 0.983 |
| CLIP, LoRA | 0.665 | 0.839 | 0.914 | 0.956 | 0.975 | 0.980 |
| CLIP, partial FT | 0.583 | 0.816 | 0.906 | 0.949 | 0.972 | 0.978 |
| CLIP, linear probe | 0.529 | 0.761 | 0.850 | 0.908 | 0.931 | 0.942 |
| supervised, full FT | 0.602 | 0.771 | 0.910 | 0.958 | 0.978 | 0.981 |
| supervised, LoRA | 0.602 | 0.802 | 0.913 | 0.959 | 0.975 | 0.980 |
| supervised, partial FT | 0.573 | 0.777 | 0.897 | 0.953 | 0.976 | 0.982 |
| supervised, linear probe | 0.497 | 0.706 | 0.796 | 0.901 | 0.940 | 0.946 |

**Verdicts** (tuned experiments; details and per-seed values in section 7 of the notebook)

| | verdict | evidence |
|---|---|---|
| H1 | **confirmed for LoRA, rejected for the linear probe** | In the 12 low-shot cells (3 datasets x 2 backbones x 1 and 5 images per class) LoRA is never below full FT beyond the tie threshold, and clearly better in 5 (e.g. CLIP at 1 image/class: -0.031 on EuroSAT and Pets). The linear probe is clearly below full FT in 11 of 12. |
| H2 | **catch-up observed, no crossover** | CLIP on EuroSAT: full FT - LoRA goes from -0.031 at 1 image/class to between +0.002 and +0.010 from 20 up to 1,500 images/class, within the seed noise also with 2-4x more training steps. The gap to the linear probe shrinks (0.106 to 0.041) but never closes. |
| H3 | **dataset-dependent** | CLIP is far better on Flowers102 (+0.28 to +0.39 at 1 image/class) and better on EuroSAT at 5-20 images/class (+0.036 to +0.065 at 5; robust to the choice of selection seed only for full FT and partial FT), with no difference above 0.01 from 100 images/class on; the supervised backbone is better on Pets at 1 image/class (-0.18 to -0.26). A per-class analysis on Pets ties this to how densely the ImageNet-1k labels cover the domain: CLIP is ahead on the cat breeds that are not ImageNet classes, while the supervised backbone is ahead on the dog breeds, including the three that are not ImageNet classes. |

The replication of CLIP-LoRA (text-initialised classifier, both encoders adapted) is within 0.6 points of the paper on 3 of the 4 trained cells and 1.3 points below on Flowers102 at 4 shots, so the low 1-shot accuracies of the main study come from the protocol (no text prior), not from the pipeline.

**Main limitations** (complete list in the notebook): three datasets and two backbones; H2 beyond 20 images/class rests on EuroSAT only, where accuracies saturate at 0.97-0.98; learning rates tuned on seed 0 with a validation set larger than the training set at low shots; 3 seeds.

## Reproducing

```bash
pip install -r requirements.txt   # Python 3.13; PyTorch with CUDA
jupyter lab low_data_finetuning.ipynb
```

- **Data (only needed for training).** Flowers102 and Oxford-IIIT Pets are downloaded by `torchvision` into `data/` on the first run. EuroSAT must be downloaded by hand: get the RGB version (`EuroSAT_RGB.zip`, 27,000 images) from the dataset page, [github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT), and extract it so that the class folders are in `data/eurosat/EuroSAT_RGB/` (`AnnualCrop/`, `Forest/`, ...). `data/en_classnames.json` (class names of the CLIP benchmark, used by the replication and the overlap check) is part of the repository.
- **Weights.** The pre-trained backbones are downloaded from the Hugging Face Hub by `timm` and `transformers` on the first run.
- **Re-executing the notebook.** The notebook presents the results: sections 3-7 only read the run logs in `results/` (one CSV row per finished run), so *Run all* rebuilds every table and figure in seconds and needs neither the datasets nor a GPU (only `data/en_classnames.json`).
- **Recomputing from scratch.** All the training code is in one cell in the Appendix, committed with `RUN_TRAINING = False`. Move the CSVs out of `results/`, set `RUN_TRAINING = True` and run the notebook: every configuration missing from the logs is trained and appended, so the run can be interrupted and resumed. About 24 hours of recorded run time on one 8 GB laptop GPU (RTX PRO 2000); on a GPU of this size run one process at a time.

## Repository layout

```
low_data_finetuning.ipynb   the report with its analyses: problem, hypotheses, data, methods, experiments, ablations, discussion,
                            references; training code in the Appendix (not executed)
requirements.txt            pinned Python dependencies
data/en_classnames.json     class names of the CLIP benchmark (the datasets themselves are not committed)
results/                    one CSV row per finished run
  results.csv                                   Experiment 1 (main study)
  replication.csv                               Experiment 2, CLIP-LoRA with the official LoRA scaling
  replication_scale_alpha_over_r.csv            Experiment 2, alpha/r scaling (sensitivity check)
  replication_zeroshot_coop_prompts.csv         Experiment 2, zero-shot CLIP with CoOp's prompts
  scaling_eurosat.csv, scaling_eurosat_vit_b16_sup.csv            Experiment 3 (CLIP, supervised)
  tuned_flowers102.csv, tuned_pets.csv          Experiment 4
  ablation.csv                                  6.1 ablations
  budget_check_eurosat.csv                      6.2 training budget
  scaling_eurosat_ext_clip_vitb16.csv, scaling_eurosat_ext_vit_b16_sup.csv   6.3 Check 2, extended grid
  lbfgs_probe.csv                               6.4 logistic-regression probe
```

## Main references

- Radford, A., et al. Learning transferable visual models from natural language supervision (CLIP). ICML, 2021.
- Zanella, M., Ben Ayed, I. Low-rank few-shot adaptation of vision-language models (CLIP-LoRA). arXiv:2405.18541, 2024.
- Zhou, K., et al. Learning to prompt for vision-language models (CoOp). IJCV, 2022.
- Hu, E. J., et al. LoRA: Low-rank adaptation of large language models. ICLR, 2022.
- Dosovitskiy, A., et al. An image is worth 16x16 words (ViT). ICLR, 2021; Steiner, A., et al. How to train your ViT? TMLR, 2022.

The full reference list is in section 8 of the notebook.
