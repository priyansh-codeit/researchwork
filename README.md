
# Photovoltaic Power Nowcasting Using Ambient Sensor Data
### An Evaluation of LightGBM on the OPERA/UniVer Benchmark

Code and manuscript for a study of photovoltaic (PV) output-power nowcasting from ambient sensor data. Using the openly released OPERA feature files of Almonacid-Olleros et al. (*Sensors*, 2020), we train regularised LightGBM models and compare them, descriptively, with the benchmark's best Random Forest.

**Authors:** Saumya Sinha, Priyansh Yadav, Chirag Mishra, Dr. Rajesh Mahadeva, Dr. Vijay Babu Koreboina
**Affiliation:** School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, Karnataka, India

---

## Summary

- **Task:** nowcast PV output power (`Pa1`) from concurrently measured irradiance, ambient temperature and module temperature. Current, voltage and power-derived columns are never used as inputs.
- **Data:** 23,984 aligned samples, 9 June to 23 November 2019, from the grid-connected UniVer generator at the University of Jaén (benchmark description: 220 multicrystalline-silicon modules, 59.4 kW nominal).
- **Features:** 36 environmental window statistics (mean, max, min, std of `Rad`, `Tamb`, `Tmod` over three incremental windows), plus six engineered irradiance–temperature interaction and irradiance-change features (42 in total). No calendar or timestamp features.
- **Models:** four LightGBM candidates (environmental-only / engineered × squared-error / absolute-error objective), selected by validation RMSE.
- **Protocol:** random 80/20 holdout, with a further random 80/20 validation split inside the training set for early stopping and model selection.

### Main result (random 80/20 holdout, N = 4,797)

| Metric | This study (LightGBM) | Benchmark Random Forest (30-fold CV) | Change |
|---|---|---|---|
| MAE | 140.21 W | 173.47 W | 19.2% lower |
| RMSE | 287.45 W | 360.13 W | 20.2% lower |
| R² | 0.998929 | 0.9983 | n/a (saturates near 1) |

> **Important caveat.** The two figures come from different evaluation protocols (single random holdout versus the benchmark's 30-fold cross-validation, whose folds are not specified). Overlapping windows mean random splits can be optimistic. The comparison is **descriptive, not protocol-matched**, and no chronological (forward-in-time) test has been run. See the *Limitations* section of the paper.

### Validation-subset comparison of the four candidates

| Features | Objective | MAE (W) | RMSE (W) | Selected |
|---|---|---|---|---|
| Environmental-only | Squared error | 151.61 | 316.48 | |
| Environmental-only | Absolute error | 155.57 | 362.28 | |
| **Engineered** | **Squared error** | **147.31** | **311.55** | ✓ |
| Engineered | Absolute error | 151.57 | 348.03 | |

On the validation subset the engineered features lower MAE by 2.6–2.8% and RMSE by 1.6–3.9%. These values were used for model selection and are not independent test estimates.

---

## Repository contents

```
.
├── sensors_impr_code_final.ipynb   # full pipeline (loading, alignment, selection, evaluation)
├── main.tex                        # manuscript source
├── references.bib                  # bibliography
├── figures/                        # figures used in the manuscript
├── opera_dataset_1w_10min.csv      # OPERA data (download separately, see below)
├── opera_dataset_3w_90min.csv      # OPERA data (download separately, see below)
└── pv_lightgbm_minimal_results/    # created when the pipeline runs
```

The code is a single self-contained script (kept in the notebook). The OPERA CSV files are not redistributed here.

---

## Data

Download the two files from the benchmark authors' repository: <https://github.com/galmonacid/opera/>

| File | Rows | Content |
|---|---|---|
| `opera_dataset_1w_10min.csv` | 24,031 | single 10-minute window |
| `opera_dataset_3w_90min.csv` | 24,024 | three incremental windows (10 + 20 + 60 min) |

Place both files in the same folder as the script. Each file must contain a `timestamp` column and the target `Pa1`.

---

## Installation

Python 3.9 or later is recommended.

```bash
pip install numpy pandas scikit-learn lightgbm joblib
```

---

## Running the pipeline

Run the notebook top to bottom, or export it as a script and run:

```bash
python sensors_impr_code_final.py
```

Input and output locations are the defaults defined near the top of the script:

```python
DEFAULT_10M = ROOT / "opera_dataset_1w_10min.csv"
DEFAULT_90M = ROOT / "opera_dataset_3w_90min.csv"
DEFAULT_OUT = ROOT / "pv_lightgbm_minimal_results"
```

> **Note:** `main()` calls `parser.parse_args(args=[])` so that it runs inside Colab. As a result the command-line flags `--data-10m`, `--data-90m` and `--out` are currently ignored. To use other paths, edit the three constants above, or change that line to `parser.parse_args()` when running from a terminal.

### What the pipeline does

1. **Load and clean** both CSVs, dropping rows with invalid timestamp or target and rejecting duplicate timestamps.
2. **Check alignment:** `Pa1` must agree between the two files at all matching timestamps (tolerance 10⁻³ W). Keep only rows whose trailing nine 10-minute timestamps are complete and consecutive, leaving **23,984** samples.
3. **Select environmental predictors only.** Column names indicating power, current, voltage, output, energy, target, etc. are rejected.
4. **Build two feature sets:** `env_only` (36 columns) and `env_engineered` (42 columns).
5. **Split:** random 80/20 outer split (seed 42) into training (19,187) and test (4,797); the training part is split again (seed 43) into fitting (15,349) and validation (3,838).
6. **Select the model:** fit the four candidates on the fitting part with early stopping (patience 60 rounds) on the validation subset, and choose the lowest validation RMSE.
7. **Refit and test:** refit the selected configuration on the full outer training set, with the number of trees set to the selected iteration, and evaluate **once** on the untouched test set.

### Engineered features

With `R₁`, `T_m,1`, `T_a,1` the first-window mean irradiance, module temperature and ambient temperature, and `R₂`, `R₃` the mean irradiance in the second and third windows:

`R₁²`, `R₁·T_m,1`, `R₁·T_a,1`, `T_m,1 − T_a,1`, `R₁ − R₂`, `R₁ − R₃`

### Model configuration (all candidates)

| Parameter | Value |
|---|---|
| `num_leaves` | 31 |
| `n_estimators` (maximum) | 1000 |
| `learning_rate` | 0.025 |
| `min_child_samples` | 25 |
| `subsample` / `colsample_bytree` | 0.90 / 0.90 (`subsample_freq=1`) |
| `reg_alpha` (L1) / `reg_lambda` (L2) | 0.05 / 3.0 |
| `random_state` | 42 |
| Objectives | `regression`, `regression_l1` |

---

## Outputs

Written to `pv_lightgbm_minimal_results/`:

| File | Content |
|---|---|
| `random_validation_search.csv` | validation metrics, best iteration and fit time for each of the four candidates |
| `random_comparison.csv` | test R², MAE, RMSE and the change against the benchmark values |
| `random_test_predictions.csv` | timestamp, measured `Pa1` and prediction for every test sample |
| `random_selected_feature_importance.csv` | gain and split importance of the selected model |
| `random_recipe.json` | seeds, selected configuration, row counts, validation and test metrics, and a protocol warning |
| `selected_lightgbm_model.joblib` | the fitted model, feature column list and final configuration |

---

## Reproducibility

- Seeds are fixed: 42 for the outer split and the models, 43 for the validation split.
- Exact numbers can vary slightly with the LightGBM and scikit-learn versions and the number of threads; record your versions when comparing.
- Test labels are never used for early stopping or model selection.

---

## Limitations

These are discussed in detail in the paper:

- **Random holdout.** About 96% of test samples have a training sample exactly 10 minutes away, so the holdout mostly measures interpolation between neighbouring windows.
- **No protocol-matched comparison.** The benchmark's 30-fold cross-validation could not be reproduced, and no Random Forest was trained on the same split.
- **Single seed, single split**, with no confidence intervals.
- **Selection is not nested:** the validation subset is used for both early stopping and candidate selection.
- **One site and one season** (June–November 2019).
- **Feature contribution** was assessed on the validation subset only; no test-set ablation was run.

---

## Citation

If you use this work, please cite the manuscript:

```bibtex
@unpublished{sinha_pv_nowcasting_lightgbm,
  title  = {Photovoltaic Power Nowcasting Using Ambient Sensor Data:
            An Evaluation of LightGBM on the OPERA/UniVer Benchmark},
  author = {Sinha, Saumya and Yadav, Priyansh and Mishra, Chirag and
            Mahadeva, Rajesh and Koreboina, Vijay Babu},
  note   = {Manuscript}
}
```

Please also cite the dataset and benchmark:

> G. Almonacid-Olleros, G. Almonacid, J. I. Fernandez-Carrasco, M. Espinilla-Estevez and J. Medina-Quero, "A New Architecture Based on IoT and Machine Learning Paradigms in Photovoltaic Systems to Nowcast Output Energy," *Sensors*, vol. 20, no. 15, 4224, 2020. <https://doi.org/10.3390/s20154224>

---

## Acknowledgements

We thank Dr. Rajesh Mahadeva and Dr. Vijay Babu Koreboina for their guidance, the SolarMobil team for a collaborative environment, Manipal Institute of Technology, Manipal Academy of Higher Education, and the creators of the OPERA dataset. ChatGPT (OpenAI) and Claude (Anthropic) were used to assist with language editing, LaTeX formatting, coding and the preparation of the methodological pipeline and graphical abstract; all contributions were reviewed and edited by the authors.

## License

*Add a license for the code (for example MIT) before publishing the repository. The OPERA data remain under the terms set by their original authors.*

## Contact

Saumya Sinha (saumyasinha0706@gmail.com) · Priyansh Yadav (y.priyansh2438@gmail.com)
