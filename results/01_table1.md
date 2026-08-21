### Table I - Symmetric Relevance Gain (eq. 5), higher is better

`paper` / **`ours`** for each method, best per row in bold. Datasets are rebuilt with the Supplementary Note E preprocessing and match Table I's N/M/d. In the paper WaX is best or tied in 26 of Table I's 30 rows, with Occlusion taking most of the rest.

| dataset | N / M / d | model | MeanShift | Occlusion | Coupling | WaX | WaX best? |
|---|---|---|---|---|---|---|---|
| wisconsin | 332/163/30 | p=1, q=2 | 1.82 / **1.77** | 1.81 / 1.77 | 1.82 / 1.77 | 1.82 / 1.77 | tie |
| wisconsin | 332/163/30 | p=2, q=1 | 7.06 / 6.91 | 7.07 / 6.91 | 7.04 / 6.90 | 7.08 / **6.93** | yes |
| wisconsin | 332/163/30 | p=2, q=2 | 1.82 / 1.79 | 1.84 / 1.78 | 1.84 / **1.79** | 1.85 / 1.79 | tie |
| wisconsin | 332/163/30 | p=2, q=inf | 0.50 / 0.49 | 0.24 / 0.24 | 0.50 / 0.49 | 0.54 / **0.51** | yes |
| wisconsin | 332/163/30 | p=10, q=2 | 1.99 / 1.86 | 2.33 / 2.17 | 2.10 / 1.96 | 2.34 / **2.17** | yes |
| musk1 | 133/167/166 | p=1, q=2 | 0.94 / 0.99 | 2.19 / 2.20 | 2.19 / 2.12 | 2.19 / **2.21** | yes |
| musk1 | 133/167/166 | p=2, q=1 | 6.24 / 7.61 | 12.85 / **12.40** | 10.65 / 10.00 | 12.84 / 12.37 | no |
| musk1 | 133/167/166 | p=2, q=2 | 1.02 / 1.12 | 2.07 / 2.09 | 2.04 / **2.09** | 2.07 / 2.09 | tie |
| musk1 | 133/167/166 | p=2, q=inf | 0.31 / 0.29 | 0.89 / 0.94 | 0.57 / 0.34 | 0.97 / **1.03** | yes |
| musk1 | 133/167/166 | p=10, q=2 | 0.92 / 0.83 | 2.14 / 1.68 | 1.72 / 1.61 | 2.30 / **2.00** | yes |
| wine-quality | 3783/1086/12 | p=1, q=2 | 1.08 / 1.08 | 1.08 / 1.08 | 1.09 / 1.07 | 1.09 / **1.08** | yes |
| wine-quality | 3783/1086/12 | p=2, q=1 | 2.58 / **2.58** | 2.55 / 2.54 | 2.56 / 2.54 | 2.58 / 2.57 | no |
| wine-quality | 3783/1086/12 | p=2, q=2 | 1.01 / 1.01 | 1.03 / 1.03 | 1.03 / **1.04** | 1.04 / 1.04 | tie |
| wine-quality | 3783/1086/12 | p=2, q=inf | 0.47 / 0.48 | 0.50 / 0.50 | 0.45 / 0.46 | 0.50 / **0.51** | yes |
| wine-quality | 3783/1086/12 | p=10, q=2 | 0.89 / 0.89 | 0.96 / 0.97 | 0.96 / 0.97 | 0.98 / **0.99** | yes |
