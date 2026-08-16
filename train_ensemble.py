#!/usr/bin/env python3
"""Train text/keyword/location BERT classifiers and build a Kaggle submission."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401 - registers TF Text ops used by TF Hub
from official.nlp import optimization

BERT_ENCODER = os.getenv(
    "BERT_ENCODER",
    "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3",
)
BERT_PREPROCESS = os.getenv(
    "BERT_PREPROCESS",
    "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
)

DEFAULT_EPOCHS = {
    "text": 5,
    "keyword": 7,
    "location": 4,
}


def build_classifier() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_text")
    preprocessing = hub.KerasLayer(BERT_PREPROCESS, name="preprocessing")
    encoder_inputs = preprocessing(text_input)
    encoder = hub.KerasLayer(BERT_ENCODER, trainable=True, name="bert_encoder")
    encoded = encoder(encoder_inputs)
    x = tf.keras.layers.Dropout(0.1)(encoded["pooled_output"])
    logits = tf.keras.layers.Dense(1, name="classifier")(x)
    return tf.keras.Model(text_input, logits)


def make_optimizer(num_examples: int, batch_size: int, epochs: int):
    steps_per_epoch = max(1, math.ceil(num_examples / batch_size))
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    return optimization.create_optimizer(
        init_lr=3e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )


def train_view(
    train_df: pd.DataFrame,
    column: str,
    view_name: str,
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> tf.keras.Model:
    subset = train_df.dropna(subset=[column]).copy()
    subset[column] = subset[column].astype(str)

    texts = subset[column].to_numpy(dtype=str)
    labels = subset["target"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    dataset = dataset.shuffle(
        min(len(subset), 4096), seed=42, reshuffle_each_iteration=True
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_classifier()
    optimizer = make_optimizer(len(subset), batch_size, epochs)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
    )

    print(
        f"Training {view_name} model on {len(subset):,} examples "
        f"for {epochs} epochs"
    )
    model.fit(dataset, epochs=epochs)

    model_path = output_dir / view_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, include_optimizer=False)
    return model


def predict_view(
    model: tf.keras.Model,
    values: pd.Series,
    batch_size: int,
) -> np.ndarray:
    available = values.notna() & values.astype(str).str.strip().ne("")
    probabilities = np.full(len(values), np.nan, dtype=np.float32)

    if not available.any():
        return probabilities

    texts = values.loc[available].astype(str).to_numpy(dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices(texts).batch(batch_size)
    logits = model.predict(dataset, verbose=0).reshape(-1)
    probabilities[available.to_numpy()] = tf.sigmoid(logits).numpy()
    return probabilities


def majority_vote(
    text_prob: np.ndarray,
    keyword_prob: np.ndarray,
    location_prob: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros(len(text_prob), dtype=np.int32)

    for i in range(len(predictions)):
        text_vote = int(text_prob[i] >= 0.5)
        votes = [text_vote]

        if np.isfinite(keyword_prob[i]):
            votes.append(int(keyword_prob[i] >= 0.5))
        if np.isfinite(location_prob[i]):
            votes.append(int(location_prob[i] >= 0.5))

        ones = sum(votes)
        zeros = len(votes) - ones
        predictions[i] = text_vote if ones == zeros else int(ones > zeros)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a three-view BERT ensemble for Kaggle disaster tweets."
    )
    parser.add_argument("--train", default="Data/train.csv")
    parser.add_argument("--test", default="Data/test.csv")
    parser.add_argument("--output-dir", default="artifacts/models")
    parser.add_argument("--submission", default="submission_ensemble.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-epochs", type=int, default=DEFAULT_EPOCHS["text"])
    parser.add_argument(
        "--keyword-epochs", type=int, default=DEFAULT_EPOCHS["keyword"]
    )
    parser.add_argument(
        "--location-epochs", type=int, default=DEFAULT_EPOCHS["location"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(42)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    required_train = {"text", "keyword", "location", "target"}
    required_test = {"id", "text", "keyword", "location"}
    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_model = train_view(
        train_df,
        "text",
        "text",
        args.text_epochs,
        args.batch_size,
        output_dir,
    )
    keyword_model = train_view(
        train_df,
        "keyword",
        "keyword",
        args.keyword_epochs,
        args.batch_size,
        output_dir,
    )
    location_model = train_view(
        train_df,
        "location",
        "location",
        args.location_epochs,
        args.batch_size,
        output_dir,
    )

    text_prob = predict_view(text_model, test_df["text"], args.batch_size)
    keyword_prob = predict_view(keyword_model, test_df["keyword"], args.batch_size)
    location_prob = predict_view(location_model, test_df["location"], args.batch_size)

    ensemble_pred = majority_vote(text_prob, keyword_prob, location_prob)
    text_pred = (text_prob >= 0.5).astype(np.int32)

    pd.DataFrame({"id": test_df["id"], "target": text_pred}).to_csv(
        "submission_text.csv", index=False
    )
    pd.DataFrame({"id": test_df["id"], "target": ensemble_pred}).to_csv(
        args.submission, index=False
    )

    print(f"Saved text baseline to submission_text.csv")
    print(f"Saved multi-view ensemble to {args.submission}")


if __name__ == "__main__":
    main()
