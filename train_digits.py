import argparse
from pathlib import Path
import tensorflow as tf

# Build a simple CNN for 32x32 grayscale digit classification (classes: 1..9).
def build_model(img_size: int = 32, num_classes: int = 9, invert: bool = False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    if invert:
        x = tf.keras.layers.Lambda(lambda t: 1.0 - t, name="invert")(x)

    x = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="digit_cnn_32x32")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Create train/val datasets from directory structure: data_dir/1 ... data_dir/9
def make_datasets(data_dir: Path, img_size: int, batch_size: int, val_split: float, seed: int):
    class_names = [str(i) for i in range(1, 10)]
    missing = [c for c in class_names if not (data_dir / c).exists()]
    if missing:
        raise SystemExit(f"[ERROR] Missing class folder(s): {missing}. Expected: {class_names} under {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
        class_names=class_names,  # <-- corrected argument name
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="validation",
        class_names=class_names,  # <-- corrected argument name
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def main():
    parser = argparse.ArgumentParser(description="Train a 32x32 grayscale digit CNN for Sudoku (classes 1..9)." )
    parser.add_argument("--data-dir", type=str, default="original", help="Path to dataset dir with subfolders 1..9")
    parser.add_argument("--img-size", type=int, default=32, help="Target square size (default: 32)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337)")
    parser.add_argument("--invert", action="store_true", help="Invert grayscale inside model (if digits are black-on-white)")
    parser.add_argument("--out-model", type=str, default="digit_cnn.keras", help="Path to save full Keras model (.keras)")
    parser.add_argument("--out-weights", type=str, default="digit_cnn.weights.h5", help="Path to save weights only (.h5)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] Data directory not found: {data_dir.resolve()}")

    # Optional: GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print(f"[INFO] Loading datasets from: {data_dir.resolve()}")
    train_ds, val_ds, class_names = make_datasets(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"[INFO] Classes: {class_names}")

    model = build_model(img_size=args.img_size, num_classes=9, invert=args.invert)
    model.summary()

    # Callbacks
    ckpt_dir = Path("ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_weights_path = ckpt_dir / "best.weights.h5"

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_weights_path),
                                           monitor="val_accuracy",
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    ]

    print("[INFO] Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # Load best weights and save final artifacts
    if ckpt_weights_path.exists():
        model.load_weights(str(ckpt_weights_path))

    print(f"[INFO] Saving full model to: {args.out_model}")
    model.save(args.out_model)  # .keras format

    print(f"[INFO] Saving weights to: {args.out_weights}")
    model.save_weights(args.out_weights)

    # Save class names for auditing
    with open("class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    print("[DONE] Training complete. Artifacts:")
    print(f"  - Full model: {args.out_model}")
    print(f"  - Weights:    {args.out_weights}")
    print(f"  - Best ckpt:  {ckpt_weights_path}")
    print("Use load_model(<.keras>) OR build_model(); load_weights(<.h5>) in your app.")

if __name__ == "__main__":
    main()
