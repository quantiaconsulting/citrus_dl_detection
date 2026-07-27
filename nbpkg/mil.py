"""
mil.py — Attention-based Multiple-Instance Learning for fruit-level infestation.

A fruit (bag) holds its 72 slices (instances). A frozen ImageNet backbone maps
each slice to a feature vector; a gated-attention head (Ilse, Tomczak & Welling,
2018) learns which slices matter and produces one probability per fruit. No slice
cleaning and no voting threshold are needed, and the attention weights show which
slices drove each decision.

Efficiency: per-slice features are extracted ONCE per backbone (frozen) and cached
to disk, so training the small attention head is fast.

Design notes
------------
* Bags are fixed size (72 slices), so no masking is required.
* The backbone is frozen (feature extractor). Fine-tuning the last blocks is a
  possible next step if the frozen features underfit the fine galleries.
* Split is by fruit (from notebook 01); each fruit is one independent bag.
"""
from __future__ import annotations
import importlib
from pathlib import Path

import numpy as np

import dataset as ds

_PREPROCESS_MODULE = {
    "EfficientNetB0": "tensorflow.keras.applications.efficientnet",
    "MobileNetV2":    "tensorflow.keras.applications.mobilenet_v2",
    "ResNet50":       "tensorflow.keras.applications.resnet50",
    "NASNetMobile":   "tensorflow.keras.applications.nasnet",
    "DenseNet121":    "tensorflow.keras.applications.densenet",
    "InceptionV3":    "tensorflow.keras.applications.inception_v3",
}


# --------------------------------------------------------------------------- #
# Feature extraction (cached)
# --------------------------------------------------------------------------- #
def _feature_extractor(cfg, backbone):
    import tensorflow as tf
    base = getattr(tf.keras.applications, backbone)(
        include_top=False, weights="imagenet",
        input_shape=(cfg.img_size, cfg.img_size, 3), pooling="avg")
    base.trainable = False
    return base


def _scan_batch(scan_dir, cfg, backbone):
    import tensorflow as tf
    preprocess = importlib.import_module(_PREPROCESS_MODULE[backbone]).preprocess_input
    paths = ds.slice_paths(scan_dir)
    sz = cfg.img_size

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (sz, sz))
        img = tf.image.grayscale_to_rgb(img)
        return preprocess(tf.cast(img, tf.float32))

    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(cfg.batch_size)), len(paths)


def extract_features(cfg, backbone, fruits, *, cache_path=None, verbose=True):
    """Return {fruit_id: (n_slices, D) float32}. Cached to `cache_path` (.npz)."""
    cache_path = Path(cache_path) if cache_path else \
        Path(cfg.out_dir) / f"features_{backbone}.npz"
    if cache_path.exists():
        if verbose:
            print(f"[{backbone}] loading cached features from {cache_path.name}")
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    if verbose:
        print(f"[{backbone}] extracting features for {len(fruits)} fruits ...")
    base = _feature_extractor(cfg, backbone)
    feats = {}
    for i, f in enumerate(fruits):
        # one fruit == one scan under fruit_key='scan'
        ds_, n = _scan_batch(f.scans[0].directory, cfg, backbone)
        feats[f.fruit_id] = base.predict(ds_, verbose=0).astype("float32")
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(fruits)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **feats)
    if verbose:
        print(f"[{backbone}] saved {cache_path.name}")
    return feats


def stack_bags(feats, fruit_ids):
    """(N, n_slices, D) array in the given fruit order."""
    return np.stack([feats[i] for i in fruit_ids]).astype("float32")


# --------------------------------------------------------------------------- #
# Attention-MIL model
# --------------------------------------------------------------------------- #
def build_attention_mil(n_slices, feat_dim, *, attn_dim=128, hidden=64,
                        dropout=0.25, lr=1e-3):
    """Returns (train_model, infer_model) sharing weights.
    train_model has a single 'bag' output (clean single-output loss);
    infer_model additionally returns the attention weights for interpretability."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    inp = layers.Input((n_slices, feat_dim))
    x = layers.Dropout(dropout)(inp)
    V = layers.Dense(attn_dim, activation="tanh")(x)          # (n, L)
    U = layers.Dense(attn_dim, activation="sigmoid")(x)       # gated
    A = layers.Multiply()([V, U])
    A = layers.Dense(1)(A)                                    # (n, 1)
    A = layers.Softmax(axis=1, name="attention")(A)           # weights over slices
    z = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                      name="bag_embedding")([A, inp])          # (D,)
    h = layers.Dense(hidden, activation="relu")(z)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(1, activation="sigmoid", name="bag")(h)
    train_model = Model(inp, out)
    train_model.compile(optimizer=optimizers.Adam(lr),
                        loss="binary_crossentropy",
                        metrics=[tf.keras.metrics.AUC(name="auc")])
    infer_model = Model(inp, [out, A])
    return train_model, infer_model


def train_mil(cfg, Xtr, ytr, Xva, yva, *, epochs=100, patience=10,
              attn_dim=128, dropout=0.25, lr=1e-3, verbose=2):
    """Train the attention head; returns the inference model (bag prob + attention)."""
    from tensorflow.keras.callbacks import EarlyStopping
    n_slices, D = Xtr.shape[1], Xtr.shape[2]
    train_model, infer_model = build_attention_mil(
        n_slices, D, attn_dim=attn_dim, dropout=dropout, lr=lr)
    es = EarlyStopping(monitor="val_auc", mode="max",
                       patience=patience, restore_best_weights=True)
    train_model.fit(Xtr, ytr.astype("float32"),
                    validation_data=(Xva, yva.astype("float32")),
                    epochs=epochs, batch_size=16, callbacks=[es], verbose=verbose)
    return infer_model


def predict_bags(model, X):
    """Return (probs (N,), attention (N, n_slices))."""
    out = model.predict(X, verbose=0)
    probs = np.asarray(out[0]).ravel()
    attn = np.asarray(out[1]).reshape(X.shape[0], X.shape[1])
    return probs, attn


# --------------------------------------------------------------------------- #
# End-to-end fine-tuned MIL (backbone trainable, no cached features)
# --------------------------------------------------------------------------- #
def make_bag_dataset(fruits, cfg, backbone, labels=None, *, training=False):
    """tf.data yielding (bag_images (n_slices,H,W,3), label). One element per fruit.
    Slice paths are resolved in Python (exactly n_slices per fruit); order is
    irrelevant for attention-MIL."""
    import tensorflow as tf
    preprocess = importlib.import_module(_PREPROCESS_MODULE[backbone]).preprocess_input
    sz = cfg.img_size
    n = cfg.slices_per_fruit

    # resolve paths in Python (not in the tf graph): list[list[str]] of length n
    bag_paths, ys = [], []
    for k, f in enumerate(fruits):
        paths = ds.slice_paths(f.scans[0].directory)[:n]
        if len(paths) != n:                      # pad by repeating last (rare)
            paths = paths + [paths[-1]] * (n - len(paths))
        bag_paths.append(paths)
        ys.append(int(f.label) if labels is None else int(labels[k]))
    bag_paths = tf.constant(bag_paths)           # (N, n) string
    ys = tf.constant(ys, dtype=tf.float32)

    def _one(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (sz, sz))
        img = tf.image.grayscale_to_rgb(img)
        return preprocess(tf.cast(img, tf.float32))

    def _load_bag(paths, label):
        imgs = tf.map_fn(_one, paths, fn_output_signature=tf.float32)
        imgs.set_shape((n, sz, sz, 3))
        return imgs, label

    ds_ = tf.data.Dataset.from_tensor_slices((bag_paths, ys))
    if training:
        ds_ = ds_.shuffle(len(fruits), seed=cfg.seed)
    ds_ = ds_.map(_load_bag, num_parallel_calls=tf.data.AUTOTUNE)
    batch = getattr(cfg, "ft_batch_bags", 2)
    return ds_.batch(batch).prefetch(tf.data.AUTOTUNE)


def _set_finetune_trainable(base, n_last=20):
    """Unfreeze the last `n_last` non-BatchNorm layers; keep all BatchNorm frozen
    (so they stay in inference mode with tiny batches)."""
    import tensorflow as tf
    BN = tf.keras.layers.BatchNormalization
    base.trainable = True
    non_bn = [l for l in base.layers if not isinstance(l, BN)]
    unfreeze = set(id(l) for l in non_bn[-n_last:])
    for l in base.layers:
        l.trainable = (not isinstance(l, BN)) and (id(l) in unfreeze)


def build_finetune_mil(cfg, backbone, *, n_slices, attn_dim=128, hidden=64,
                       dropout=0.4, lr=1e-4, n_last=20):
    """End-to-end attention-MIL: TimeDistributed backbone -> gated attention -> bag.
    Backbone's last `n_last` non-BN layers are trainable; returns (train, infer)."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    sz = cfg.img_size
    base = getattr(tf.keras.applications, backbone)(
        include_top=False, weights="imagenet", input_shape=(sz, sz, 3), pooling="avg")
    _set_finetune_trainable(base, n_last=n_last)

    bag_in = layers.Input((n_slices, sz, sz, 3))
    feats = layers.TimeDistributed(base)(bag_in)             # (n_slices, D)
    x = layers.Dropout(dropout)(feats)
    V = layers.Dense(attn_dim, activation="tanh")(x)
    U = layers.Dense(attn_dim, activation="sigmoid")(x)
    A = layers.Multiply()([V, U])
    A = layers.Dense(1)(A)
    A = layers.Softmax(axis=1, name="attention")(A)
    z = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                      name="bag_embedding")([A, feats])
    h = layers.Dense(hidden, activation="relu")(z)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(1, activation="sigmoid", name="bag")(h)
    train_model = Model(bag_in, out)
    train_model.compile(optimizer=optimizers.Adam(lr), loss="binary_crossentropy",
                        metrics=[tf.keras.metrics.AUC(name="auc")])
    infer_model = Model(bag_in, [out, A])
    return train_model, infer_model


def train_finetune_mil(cfg, backbone, train_fruits, val_fruits, *,
                       epochs=40, patience=8, lr=1e-4, n_last=20, verbose=2):
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping
    train_model, infer_model = build_finetune_mil(
        cfg, backbone, n_slices=cfg.slices_per_fruit, lr=lr, n_last=n_last)
    tr = make_bag_dataset(train_fruits, cfg, backbone, training=True)
    va = make_bag_dataset(val_fruits, cfg, backbone, training=False)
    # balanced class weights (78 infested vs 27 control -> don't collapse to infested)
    y = np.array([f.label for f in train_fruits]); n = len(y)
    cw = {0: n / (2 * max((y == 0).sum(), 1)), 1: n / (2 * max((y == 1).sum(), 1))}
    es = EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                       restore_best_weights=True)
    train_model.fit(tr, validation_data=va, epochs=epochs, class_weight=cw,
                    callbacks=[es], verbose=verbose)
    return infer_model


def predict_bags_ft(infer_model, fruits, cfg, backbone):
    """Predict probs + attention for a list of fruits (end-to-end model)."""
    import numpy as np
    ds_ = make_bag_dataset(fruits, cfg, backbone, training=False)
    out = infer_model.predict(ds_, verbose=0)
    probs = np.asarray(out[0]).ravel()
    attn = np.asarray(out[1]).reshape(len(fruits), cfg.slices_per_fruit)
    return probs, attn
