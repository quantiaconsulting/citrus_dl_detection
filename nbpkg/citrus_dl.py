"""
citrus_dl.py — TensorFlow layer built on the corrected dataset model (dataset.py).

Invariants that remove the reviewer's inconsistencies:
  * split unit = PHYSICAL FRUIT (cohort,dose,rep); all its day-scans stay together;
  * training pools a fruit's slices across all its days (curated tree if provided);
  * evaluation is PER SCAN (fruit x day); each scan carries group=fruit_id so CIs
    are cluster-bootstrapped by fruit;
  * early stopping monitors validation AUC; vote threshold chosen on validation only.
"""
from __future__ import annotations
import importlib
import numpy as np
from pathlib import Path

import eval_core as ec
import dataset as ds

_PREPROCESS_MODULE = {
    "EfficientNetB0": "tensorflow.keras.applications.efficientnet",
    "MobileNetV2":    "tensorflow.keras.applications.mobilenet_v2",
    "ResNet50":       "tensorflow.keras.applications.resnet50",
    "NASNetMobile":   "tensorflow.keras.applications.nasnet",
    "DenseNet121":    "tensorflow.keras.applications.densenet",
    "InceptionV3":    "tensorflow.keras.applications.inception_v3",
}


def set_seeds(seed=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def index_fruits(cfg):
    """Build the physical-fruit index from the real tree (see dataset.py)."""
    fruits, unparsed = ds.build_fruit_index(cfg.data_root, fruit_key=cfg.fruit_key)
    return fruits


def _curated_dir(cfg, scan_dir: Path):
    """Map an eval scan folder to its curated counterpart, if a curated_root is set."""
    if cfg.curated_root is None:
        return scan_dir
    rel = Path(scan_dir).relative_to(cfg.data_root)
    cand = Path(cfg.curated_root) / rel
    return cand if cand.is_dir() else scan_dir


def _training_slice_list(fruits, cfg):
    """(path, label) for TRAINING: pool every slice of every day-scan of each
    fruit, from the curated tree if configured."""
    paths, labels = [], []
    for f in fruits:
        for sc in f.scans:
            for sp in ds.slice_paths(_curated_dir(cfg, sc.directory)):
                paths.append(sp); labels.append(f.label)
    return paths, labels


def make_train_dataset(fruits, cfg, backbone, *, training=True):
    import tensorflow as tf
    preprocess = importlib.import_module(_PREPROCESS_MODULE[backbone]).preprocess_input
    paths, labels = _training_slice_list(fruits, cfg)
    if not paths:
        raise SystemExit("No training slices found — check paths/curated_root.")
    sz = cfg.img_size

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (sz, sz))
        img = tf.image.grayscale_to_rgb(img)
        img = preprocess(tf.cast(img, tf.float32))
        return img, tf.cast(label, tf.float32)

    ds_ = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds_ = ds_.shuffle(min(len(paths), 8192), seed=cfg.seed)
    ds_ = ds_.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds_.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE), len(paths)


def _scan_dataset(scan_dir, cfg, backbone):
    import tensorflow as tf
    preprocess = importlib.import_module(_PREPROCESS_MODULE[backbone]).preprocess_input
    paths = ds.slice_paths(scan_dir)          # full 72 slices (eval tree)
    sz = cfg.img_size

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (sz, sz))
        img = tf.image.grayscale_to_rgb(img)
        return preprocess(tf.cast(img, tf.float32))

    d = tf.data.Dataset.from_tensor_slices(paths).map(_load).batch(cfg.batch_size)
    return d, len(paths)


def build_model(cfg, backbone, *, lr, dropout, dense_units):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    base = getattr(tf.keras.applications, backbone)(
        include_top=False, weights="imagenet",
        input_shape=(cfg.img_size, cfg.img_size, 3), pooling=None)
    base.trainable = False
    inp = layers.Input((cfg.img_size, cfg.img_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model, base


def train_backbone(cfg, backbone, train_fruits, val_fruits, hp, *, verbose=2):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    model, base = build_model(cfg, backbone, lr=hp["lr"],
                              dropout=hp["dropout"], dense_units=hp["dense"])
    tr, _ = make_train_dataset(train_fruits, cfg, backbone, training=True)
    va, _ = make_train_dataset(val_fruits, cfg, backbone, training=False)
    es = EarlyStopping(monitor="val_auc", mode="max",
                       patience=cfg.patience, restore_best_weights=True)
    model.fit(tr, validation_data=va, epochs=cfg.max_epochs, callbacks=[es], verbose=verbose)
    base.trainable = True
    for l in base.layers[:-cfg.finetune_unfreeze_last]:
        l.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"] / 10),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(tr, validation_data=va, epochs=cfg.max_epochs, callbacks=[es], verbose=verbose)
    return model


def predict_scan_scores(model, cfg, backbone, fruits):
    """PER-SCAN prediction: one FruitScores per (fruit x day), group=fruit_id."""
    scores = []
    for f in fruits:
        for sc in f.scans:
            d, _ = _scan_dataset(sc.directory, cfg, backbone)
            probs = model.predict(d, verbose=0).ravel()
            scores.append(ec.aggregate_fruit(f"{f.fruit_id}@d{sc.day}", f.label,
                          probs, slice_decision=cfg.slice_decision, group=f.fruit_id))
    return scores


def run_optuna(cfg, backbone, train_fruits, val_fruits, *, n_trials=None):
    import optuna
    n_trials = n_trials or cfg.hpo_n_trials
    from tensorflow.keras.callbacks import EarlyStopping

    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
              "dropout": trial.suggest_float("dropout", 0.1, 0.7),
              "dense": trial.suggest_categorical("dense", [32,64,128,256,512])}
        model, _ = build_model(cfg, backbone, lr=hp["lr"],
                               dropout=hp["dropout"], dense_units=hp["dense"])
        tr, _ = make_train_dataset(train_fruits, cfg, backbone, training=True)
        va, _ = make_train_dataset(val_fruits, cfg, backbone, training=False)
        es = EarlyStopping(monitor="val_auc", mode="max",
                           patience=cfg.patience, restore_best_weights=True)
        h = model.fit(tr, validation_data=va, epochs=cfg.hpo_epochs,
                      callbacks=[es], verbose=0)
        return max(h.history["val_auc"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), study
