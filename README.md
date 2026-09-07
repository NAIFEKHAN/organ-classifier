# Organ Scan Classifier

Upload a medical scan image and the model predicts which organ it is:
**Brain, Lungs, Heart, Liver, Kidney, or Pancreas** (Gallbladder and Bladder
can be added later once suitable datasets are found — see note at the bottom).

This identifies the **organ only**, not any disease or condition.

Everything in this project has been built and tested end-to-end (with a
tiny synthetic dataset) except the two steps that need real internet access
this sandbox doesn't have: downloading the actual medical datasets (now
automated via `src/download_datasets.py` / `setup.ps1` / `setup.sh`, but
untested against the live Kaggle API from here), and downloading
pretrained ImageNet weights during training. Both work normally on your
own machine or in Google Colab. Run `download_datasets.py` on your own
machine first — if any Kaggle slug has changed, it'll tell you exactly
which organ failed instead of failing silently.

## Project structure

```
organ-classifier/
├── setup.ps1              # one-command setup (Windows PowerShell)
├── setup.sh               # one-command setup (macOS/Linux)
├── data/
│   ├── raw/              # download_datasets.py fills this, one folder per organ
│   └── processed/        # auto-generated train/val/test split
├── src/
│   ├── download_datasets.py        # downloads all 6 Kaggle datasets automatically
│   ├── preprocessing/
│   │   ├── organize_dataset.py     # splits raw images into train/val/test
│   │   ├── extract_ct_slices.py    # turns 3D CT volumes (.nii) into 2D PNG slices
│   │   └── extract_echo_frames.py  # turns echo videos into still frames (heart)
│   ├── train.py           # trains the model (transfer learning, EfficientNetB0)
│   └── evaluate.py        # accuracy, per-class metrics, confusion matrix
├── model/                 # trained model + class list get saved here
├── backend/
│   ├── main.py             # FastAPI server with a /predict endpoint
│   └── requirements.txt
└── frontend/
    ├── index.html          # upload page
    ├── style.css
    └── script.js
```

## Step 1 — Get the datasets

### Option A — automated (recommended)

One command downloads all six Kaggle datasets and splits them into
train/val/test. First, get a free Kaggle API token (one-time):

1. Log in at [kaggle.com](https://www.kaggle.com) → your profile → **Settings**.
2. Under **API**, click **Create New Token** — this shows a token starting with `KGAT_...`.
3. Either:
   - set it as an environment variable, e.g. `export KAGGLE_API_TOKEN=KGAT_xxxxxxxx`
     (Windows PowerShell: `$env:KAGGLE_API_TOKEN="KGAT_xxxxxxxx"`), or
   - save it to `~/.kaggle/access_token` (Windows: `C:\Users\<you>\.kaggle\access_token`).

   (Older accounts can instead use "Legacy API Credentials" → `kaggle.json`, placed at
   `~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars — both still work.)

Then, from the project root:

```powershell
# Windows PowerShell
.\setup.ps1
```
```bash
# macOS / Linux
chmod +x setup.sh && ./setup.sh
```

This runs `pip install -r requirements.txt`, downloads and organizes all
six datasets, and splits them into `data/processed/`. If you'd rather run
the download step on its own (e.g. only some organs, or a size cap):

```bash
python src/download_datasets.py                       # all 6 organs
python src/download_datasets.py --organs brain lungs   # just these
python src/download_datasets.py --max_per_organ 2000   # cap images per organ
```

Datasets used (all are plain JPG/PNG images, so no volume/video
conversion is needed for these):

| Organ | Kaggle dataset | Source |
|---|---|---|
| Brain | Brain Tumor MRI Dataset | `masoudnickparvar/brain-tumor-mri-dataset` |
| Lungs | Chest X-Ray Images (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` |
| Heart | CAMUS echocardiography images | `toygarr/camus-dataset` |
| Liver | LiTS liver CT, pre-sliced to PNG | `andrewmvd/lits-png` |
| Kidney | CT KIDNEY Dataset (Normal-Cyst-Tumor-Stone) | `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone` |
| Pancreas | Pancreatic CT Images | `jayaprakashpondy/pancreatic-ct-images` |

Since this project only classifies the organ (not disease), the script
dumps every disease/subtype subfolder (tumor, cyst, stone, normal, etc.)
together into that organ's `data/raw/<organ>/` folder, and skips any
segmentation-mask images automatically. Kaggle datasets occasionally get
renamed or restructured — if one organ comes back with 0 images, the
script tells you which, and you can check that dataset's page or swap in
a different slug in `src/download_datasets.py`'s `DATASETS` dict.

### Option B — manual download

Prefer to pick your own datasets, or one of these got taken down? Search
Kaggle / TCIA yourself and drop images into `data/raw/<organ_name>/`
(e.g. `data/raw/brain/`, `data/raw/lungs/`, etc.) — ignoring disease
labels the same way as above. If a dataset comes as 3D volumes or
videos instead of images, convert first:

```bash
# Example: liver CT volumes → 2D slices
python src/preprocessing/extract_ct_slices.py \
    --volume_dir path/to/downloaded/liver_volumes \
    --out_dir data/raw/liver \
    --organ_label liver

# Example: heart echo videos → still frames
python src/preprocessing/extract_echo_frames.py \
    --video_dir path/to/downloaded/echo_videos \
    --out_dir data/raw/heart
```

Then split into train/val/test:

```bash
pip install -r requirements.txt
python src/preprocessing/organize_dataset.py \
    --raw_dir data/raw --out_dir data/processed
```

## Step 2 — Train the model (use Google Colab for a free GPU)

1. Open [Google Colab](https://colab.research.google.com), enable a GPU runtime
   (Runtime → Change runtime type → GPU).
2. Upload the `organ-classifier` folder (or just `src/` and your prepared
   `data/processed/` folder — you can zip and upload, or use Google Drive).
3. Run:
   ```bash
   pip install -r requirements.txt
   python src/train.py --data_dir data/processed --model_out model \
       --img_size 224 --batch_size 32 --epochs_head 10 --epochs_finetune 10
   ```
4. Download the resulting `model/` folder (`organ_classifier.keras`,
   `class_names.json`, `model_config.json`) back into this project's `model/` folder.

Training time depends on dataset size — with a few thousand images per class
on a Colab GPU, expect roughly 20–40 minutes total for both training phases.

## Step 3 — Evaluate

```bash
python src/evaluate.py --data_dir data/processed \
    --model_path model/organ_classifier.keras \
    --class_names model/class_names.json
```

This prints per-class precision/recall/F1 and saves `confusion_matrix.png` —
check it to see which organs the model confuses most (liver/kidney/pancreas,
all being similar-looking CT slices, are the ones to watch).

## Step 4 — Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/health` to confirm the model loaded. FastAPI
also gives you free interactive API docs at `http://localhost:8000/docs`.

## Step 5 — Run the frontend

Simplest option — just open the file directly:
```bash
open frontend/index.html        # macOS
start frontend/index.html       # Windows
```

Or serve it locally (avoids some browsers' restrictions on local files):
```bash
cd frontend
python3 -m http.server 8080
```
then visit `http://localhost:8080`.

With the backend running on port 8000, upload an image and click **Analyze
scan** — you'll see the predicted organ and a confidence percentage for
every class.

## Adding Gallbladder and Bladder later

These two are deferred because public datasets are scarce. When you're
ready to add them:
1. Search TCIA and Mendeley Data (not just Kaggle) for ultrasound
   gallbladder / CT or MRI bladder datasets.
2. Add `data/raw/gallbladder/` and `data/raw/bladder/` folders.
3. Re-run `organize_dataset.py` and `train.py` — the training script
   automatically adapts to however many class folders it finds, so no code
   changes are needed.

## Notes

- This is a research/project demo, not a diagnostic tool — no medical
  claims should be made from its output.
- The confidence percentage shown per prediction is the model's softmax
  probability, not a measure of clinical accuracy.
- CORS is wide open (`allow_origins=["*"]`) in `backend/main.py` for local
  development — tighten this before deploying anywhere public.
