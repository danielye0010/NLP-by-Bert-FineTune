# Multi-View BERT Ensemble for Disaster Tweet Classification

A TensorFlow/BERT NLP pipeline for Kaggle's **Natural Language Processing with Disaster Tweets** challenge, using three complementary text views — **tweet content, keyword metadata, and location metadata** — and combining their predictions with modality-aware majority voting.

Instead of treating the task as a single text-classification problem, the project fine-tunes separate BERT classifiers on different information channels and fuses them at inference time.

## Highlights

- fine-tuned BERT binary classifier for full tweet text
- separate BERT models for keyword and location metadata
- independent optimization schedule for each information channel
- TensorFlow Hub preprocessing + BERT encoder
- AdamW with warmup
- missing-metadata-aware ensemble voting
- automatic Kaggle submission generation
- included Kaggle train/test data and historical model artifacts

## Architecture

The project trains three models with the same BERT classification head:

### 1. Tweet-text model
Uses the complete tweet as the primary semantic signal.

### 2. Keyword model
Fine-tunes BERT on the competition's keyword metadata, capturing compact disaster-related concepts such as event type or topic cues.

### 3. Location model
Uses the location field as an additional contextual signal when location metadata is available.

Each model applies:

```text
raw string
  -> TensorFlow Hub BERT preprocessing
  -> trainable BERT encoder
  -> pooled representation
  -> dropout
  -> binary classification logit
```

## Ensemble

At inference time, the text model always provides a vote. Keyword and location models contribute when their corresponding metadata exists.

The final classifier uses majority voting across the available views. If only two views are available and they disagree, the tweet-text model acts as the tie breaker.

This keeps the strongest semantic signal as the backbone while still allowing metadata-specific models to influence predictions.

## Repository structure

- `train_ensemble.py` — maintained end-to-end training and inference pipeline
- `disaster_tweets.py` — compatibility entry point that runs `train_ensemble.py`
- `Data/` — Kaggle train/test/sample-submission files
- `Models/` — historical saved text-model artifact
- `Model_keyword/`, `Model_location/` — historical metadata-model export artifacts
- `Bert Example/` — earlier BERT experimentation material
- `Submission Files/` — retained submission artifacts from the original project

## Installation

The maintained script follows the TensorFlow 2.11 stack used by the original project:

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for full BERT fine-tuning.

## Run

```bash
python train_ensemble.py
```

or through the compatibility entry point:

```bash
python disaster_tweets.py
```

Default inputs:

```text
Data/train.csv
Data/test.csv
```

Default outputs:

```text
artifacts/models/text/
artifacts/models/keyword/
artifacts/models/location/
submission_text.csv
submission_ensemble.csv
```

## Training configuration

The default schedule preserves the original experiment design:

| Model | Epochs |
|---|---:|
| Tweet text | 5 |
| Keyword | 7 |
| Location | 4 |

Default batch size is 64. All values can be overridden from the command line:

```bash
python train_ensemble.py \
  --batch-size 32 \
  --text-epochs 5 \
  --keyword-epochs 7 \
  --location-epochs 4
```

## Custom BERT encoder

The default model is the cased BERT-base TensorFlow Hub encoder used in the original project. Encoder and preprocessing URLs can be overridden with environment variables:

```bash
export BERT_ENCODER=<tf-hub-encoder-url>
export BERT_PREPROCESS=<tf-hub-preprocess-url>
```

## Implementation improvements

The maintained pipeline preserves the original three-view idea while making the execution consistent end to end: each model receives the same feature type during training and inference, each model gets its own optimizer/loss state, and missing keyword/location values are handled explicitly during ensemble voting.

## Contributors

- Nuo Xu
- Daniel Ye
