# Photovoltaic Power Nowcasting Using Ambient Sensor Data

### An Evaluation of LightGBM on the OPERA/UniVer Benchmark

**Authors:** Saumya Sinha, Priyansh Yadav, Chirag Mishra, Dr. Rajesh Mahadeva, Dr. Vijay Babu Koreboina
**Affiliation:** School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, Karnataka, India

---

## Overview

This repository contains the code, manuscript and supplementary material for a study of regularised **LightGBM** models for nowcasting the output power of a grid-connected photovoltaic (PV) generator from ambient sensor data (irradiance, ambient temperature, module temperature).

The study uses the public feature files released with Almonacid-Olleros et al. (*Sensors*, 2020), which reported a Random Forest with incremental 10+20+60-minute windows as the best model on a 59.4 kW generator at the University of Jaén, Spain (MAE 173.47 W, RMSE 360.13 W, 30-fold cross-validation).

The work applies an existing algorithm to a published dataset and does not claim methodological novelty. Its main aims are to:

1. Evaluate gradient boosting on the Opera benchmark.
2. Compare environmental-only features (36) with engineered features (36 + 6) on a validation subset.
3. Report a held-out test result, with model selection kept separate from testing.
4. State explicitly why the comparison with the benchmark is **not protocol-matched**.

## Key results

Random shuffled 80/20 holdout (N = 4,797 test samples), selected model = engineered features (42), squared-error objective, 31 leaves, 1,000 trees:

| Metric | LightGBM (this study) | Benchmark Random Forest (90 min) | Difference |
|---|---|---|---|
| MAE | **140.21 W** | 173.47 W | 19.2% lower |
| RMSE | **287.45 W** | 360.13 W | 20.2% lower |
| R² | 0.998929 | 0.9983 | not interpreted at the 4th decimal |

Validation-subset comparison of the four candidates:

| Features | Objective | Val. MAE (W) | Val. RMSE (W) | Selected |
|---|---|---|---|---|
| Engineered (42) | Squared error | 147.31 | 311.55 | Yes |
| Environmental-only (36) | Squared error | 151.61 | 316.48 | No |
| Engineered (42) | Absolute error | 151.57 | 348.03 | No |
| Environmental-only (36) | Absolute error | 155.57 | 362.28 | No |

Gain-based importance of the selected model: 93.0% environmental window features, 7.0% engineered features.

### Important caveats

- The benchmark used 30-fold cross-validation (shuffling not specified); this study uses a **single random holdout**. The comparison is **descriptive only**.
- Windows overlap heavily, and 96.3% of test samples have a training sample exactly 10 minutes away, so the random holdout mostly measures **interpolation** between neighbouring observations (manuscript Fig. 4).
- No chronological or blocked evaluation, nested selection, repeated seeds, confidence intervals or test-set ablation has been run.
- Single site, single season (June to November 2019).
- Test performance of the environmental-only candidates was not computed, so the feature-set comparison rests on validation data only.
- No Random Forest was trained here, so the effect of model family cannot be separated from that of the feature set.

## Repository contents

| File | Description |
|---|---|
| `sensors_impr_final_.pdf` | Main manuscript |
| `supplementary_final.pdf` | Supplementary material (inputs and alignment, feature definitions, LightGBM formulation, metrics, validation results, reproduction steps) |
| `sensors_impr_code_.ipynb` | Notebook: data loading and alignment, feature engineering, model selection, holdout evaluation, and all paper figures |

The notebook is organised as follows:

| Cells | Purpose |
|---|---|
| 0 | Locates (and, if needed, unpacks) the two Opera CSV files in the working directory |
| 1 | Main pipeline: alignment, features, 4-candidate selection, refit, single test evaluation, result export |
| 2–10 | Paper figures (dataset overview, physical structure, temporal dependence, benchmark comparison, holdout performance, example days, validation results, feature importance) |

## Data

The dataset is **not included** in this repository. Download the Opera feature files from the original release:

<https://github.com/galmonacid/opera/>

Place these two files in the working directory (or a direct sub-folder) next to the notebook:

- `opera_dataset_1w_10min.csv` (24,031 rows)
- `opera_dataset_3w_90min.csv` (24,024 rows)

Both must contain `timestamp` and `Pa1` columns.

**Alignment.** A 90-minute row is kept only if its nine trailing 10-minute timestamps are present and consecutive, giving **n = 23,984** samples (9 June to 23 November 2019). The target `Pa1` ranges from 0 to 28,472.31 W; about 26.4% of samples are below 1 W (night) and are retained.

## Method summary

**Features.** 3 signals (Rad, Tamb, Tmod) × 4 statistics (mean, max, min, std) × 3 windows (10, 20, 60 min) = **36 environmental features**. Predictors whose names indicate power, current, voltage, generation, production or energy are excluded. No calendar or timestamp features are used.

Six engineered features are added for the 42-feature set, where R₁, R₂, R₃ are the mean irradiance of windows 1 to 3 and Tₘ,₁, Tₐ,₁ are the first-window module and ambient temperatures:

`R₁²`, `R₁·Tₘ,₁`, `R₁·Tₐ,₁`, `Tₘ,₁ − Tₐ,₁`, `R₁ − R₂`, `R₁ − R₃`

**Candidates.** 2 feature sets × 2 objectives (`regression`, `regression_l1`) = 4 LightGBM models.

| Parameter | Value |
|---|---|
| Max trees | 1,000 |
| Learning rate | 0.025 |
| Leaves | 31 (no depth limit) |
| Min samples per leaf | 25 |
| Row / column subsampling | 0.90 / 0.90 (`subsample_freq=1`) |
| L2 / L1 regularisation | 3.0 / 0.05 |
| Early stopping | 60 rounds on the validation subset |

**Protocol.**

| Step | Split | Rows |
|---|---|---|
| Outer split | random, seed 42 | 19,187 train / 4,797 test |
| Inner split | random, seed 43 | 15,349 fit / 3,838 validation |

1. Fit each candidate on the fitting set with early stopping on the validation set.
2. Select the candidate with the lowest validation RMSE.
3. Refit it on all 19,187 outer-training rows using its best iteration count.
4. Evaluate **once** on the untouched test set.

## Reproducing the results

### Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `joblib`, `matplotlib`
- `jupyter` (to run the notebook)

```bash
pip install numpy pandas scikit-learn lightgbm joblib matplotlib jupyter
```

### Run

1. Download the two CSV files (see [Data](#data)) into the same folder as the notebook.
2. Open and run all cells of `sensors_impr_code_.ipynb` in order. The modelling cell must finish before the figure cells, which read its outputs.

```bash
jupyter notebook sensors_impr_code_.ipynb
```

### Outputs

Results are written to `pv_lightgbm_minimal_results/`:

| File | Content |
|---|---|
| `random_validation_search.csv` | Validation metrics for the four candidates |
| `random_test_predictions.csv` | Holdout predictions (`actual_Pa1_W`, `selected_LightGBM_prediction_W`) |
| `random_comparison.csv` | Test metrics and comparison with the published benchmark |
| `random_selected_feature_importance.csv` | Gain-based feature importance of the selected model |
| `random_recipe.json` | Run recipe (selected candidate, settings) |
| `paper_figures/` | Paper figures as `.pdf` and 300-dpi `.png` |

Exact numerical agreement with the paper depends on the same data files, library versions and seeds (outer split and model seed 42, validation split seed 43).

## Future work

Chronological and blocked (purged) evaluation; feature-group ablation; like-for-like baselines (Random Forest, XGBoost, CatBoost) on identical splits; repeated seeds and nested validation; block-bootstrap intervals; multi-season and multi-site data; quantile or conformal prediction intervals; SHAP-based error diagnosis; solar-geometry features.

## Data and code availability

- Dataset: <https://github.com/galmonacid/opera/>
- Code: <https://github.com/priyansh-codeit/researchwork>

## Citation

If you use this work, please cite the manuscript and the original benchmark:

```bibtex
@article{sinha2026lightgbm,
  title  = {Photovoltaic Power Nowcasting Using Ambient Sensor Data: An Evaluation of LightGBM on the OPERA/UniVer Benchmark},
  author = {Sinha, Saumya and Yadav, Priyansh and Mishra, Chirag and Mahadeva, Rajesh and Koreboina, Vijay Babu},
  note   = {Manuscript}
}

@article{almonacid2020opera,
  title   = {A New Architecture Based on IoT and Machine Learning Paradigms in Photovoltaic Systems to Nowcast Output Energy},
  author  = {Almonacid-Olleros, Guillermo and Almonacid, Gabino and Fernandez-Carrasco, Juan Ignacio and Espinilla-Estevez, Macarena and Medina-Quero, Javier},
  journal = {Sensors},
  volume  = {20},
  number  = {15},
  pages   = {4224},
  year    = {2020},
  doi     = {10.3390/s20154224}
}
```

## Acknowledgements

The authors thank the creators of the Opera dataset, the SolarMobil team, and Manipal Institute of Technology, Manipal Academy of Higher Education. ChatGPT (OpenAI) and Claude (Anthropic) were used to assist with language editing, LaTeX formatting, coding, and preparation of the pipeline and graphical abstract; all AI-assisted contributions were reviewed and edited by the authors.

## License

_Add a license for the code (e.g. MIT) and note that the dataset remains under the terms of its original release._
