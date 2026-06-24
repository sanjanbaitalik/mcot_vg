# Copyable Colab/Kaggle cells

## Cell 1 — Install

```python
!pip install -q torch torchvision transformers accelerate bitsandbytes>=0.46.1 datasets pillow opencv-python numpy scipy statsmodels tqdm matplotlib pyyaml grad-cam google-genai
```

## Cell 2 — Upload/unzip package

Upload `mcot_vg_accuracy_package.zip`, then run:

```python
import zipfile, os, sys
zip_path = '/content/mcot_vg_accuracy_package.zip'  # change for Kaggle if needed
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content')
os.chdir('/content/mcot_vg_accuracy_package')
!pip install -q -e .
```

## Cell 3 — Optional Gemini/Vertex credentials

```python
import os
os.environ['GCP_PROJECT'] = 'YOUR_PROJECT_ID'
os.environ['GCP_LOCATION'] = 'us-central1'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/service_account.json'
```

## Cell 4 — Run strict quick test on 20 samples

```python
!python scripts/run_full_eval.py \
  --n 20 \
  --seed 42 \
  --model_id llava-hf/llava-1.5-7b-hf \
  --methods baseline,mcot,vg_crop,mcot_vg,router \
  --judge strict \
  --out_dir outputs/debug20

!python scripts/make_report.py --out_dir outputs/debug20
```

## Cell 5 — Full Gemini semantic evaluation

```python
!python scripts/run_full_eval.py \
  --n 1145 \
  --seed 42 \
  --model_id llava-hf/llava-1.5-7b-hf \
  --methods baseline,mcot,vg_crop,mcot_vg,router \
  --judge gemini \
  --gemini_model gemini-2.5-flash \
  --out_dir outputs/llava15_v4_gemini

!python scripts/make_report.py --out_dir outputs/llava15_v4_gemini
```

## Cell 6 — Read the only claim text you should use in the paper

```python
from pathlib import Path
print(Path('outputs/llava15_v4_gemini/reports/paper_claims.md').read_text())
```

## Cell 7 — Generate human annotation CSV

```python
!python scripts/make_annotation_template.py --out_dir outputs/llava15_v4_gemini --method mcot_vg --n_items 100
```
