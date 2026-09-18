"""
breast_cancer.py
------------------
Breast Ultrasound Image Classification (BUSI) - inference wrapper.

Loads the MobileNetV2 model produced by train_model.py:
    model/breast_cancer_model.keras
    model/class_indices.json

and exposes a simple predict(image_path) -> dict API used by app.py inside
the Clinical Examination tab.

This module deliberately has no Streamlit dependency so it can be imported
and unit-tested on its own.
"""

import os
import json

import numpy as np

IMG_SIZE = (224, 224)

# Clinical framing shown alongside each predicted class in the UI / report.
CLASS_INFO = {
    "benign": {
        "label": "Benign",
        "risk": "low",
        "note": (
            "Findings are consistent with a benign lesion. Continue routine "
            "clinical follow-up."
        ),
    },
    "malignant": {
        "label": "Malignant",
        "risk": "high",
        "note": (
            "Findings are suspicious for malignancy. Urgent specialist "
            "referral and biopsy correlation are recommended."
        ),
    },
    "normal": {
        "label": "Normal",
        "risk": "none",
        "note": "No abnormal breast tissue findings detected on this scan.",
    },
}


class BreastCancerClassifier:
    """
    Thin wrapper around the trained MobileNetV2 BUSI classifier.

    Loading TensorFlow / the model file is deferred to __init__ so that a
    single instance can be created once (e.g. behind st.cache_resource) and
    reused across Streamlit reruns.
    """

    def __init__(self, model_path, class_index_path):
        self.model_path = model_path
        self.class_index_path = class_index_path
        self.model = None
        self.idx_to_class = None
        self._preprocess_input = None

        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            legacy_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "breast_cancer", "model", "breast_cancer_model.joblib")
            )
            if os.path.exists(legacy_model_path):
                self.model_path = legacy_model_path
                self._load_legacy_svm()
                return
            raise FileNotFoundError(
                f"'{self.model_path}' not found. Run `python train_model.py` "
                f"first (it saves the trained model there), then restart the app."
            )
        if not os.path.exists(self.class_index_path):
            raise FileNotFoundError(
                f"'{self.class_index_path}' not found. Run `python train_model.py` "
                f"first, then restart the app."
            )

        # Imported lazily so the rest of the app can still start up even if
        # tensorflow isn't installed / this feature isn't used.
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        self.model = tf.keras.models.load_model(self.model_path)
        self._preprocess_input = preprocess_input

        with open(self.class_index_path, "r") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    def _load_legacy_svm(self):
        """Load the supplied non-TensorFlow BUSI SVM artifact."""
        import joblib

        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self._legacy_scaler = artifact["scaler"]
        self._legacy_img_size = tuple(artifact.get("img_size", (128, 128)))
        with open(
            os.path.join(os.path.dirname(self.model_path), "class_indices.json"),
            "r",
        ) as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        self._legacy = True

    @property
    def is_ready(self):
        return self.model is not None

    def predict(self, image_path):
        """
        Run inference on a single ultrasound image.

        Returns:
            {
                "predicted_class": "benign" | "malignant" | "normal",
                "predicted_label": "Benign",
                "confidence": 0.93,
                "probabilities": {"benign": 0.93, "malignant": 0.05, "normal": 0.02},
                "risk": "low" | "high" | "none",
                "note": "...",
            }
        """
        if not self.is_ready:
            raise RuntimeError("Breast cancer model is not loaded.")

        # Defensive: coerce anything path-like (tuple/list/Path) down to a
        # plain string so tf.io.read_file gets a scalar filename.
        if isinstance(image_path, (tuple, list)):
            image_path = image_path[0]
        image_path = str(image_path)

        if getattr(self, "_legacy", False):
            from PIL import Image

            image = Image.open(image_path).convert("RGB").resize(self._legacy_img_size)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            # The supplied SVM was trained on a fixed-length image vector.
            features = pixels.reshape(-1)
            expected = self._legacy_scaler.n_features_in_
            if features.size != expected:
                features = np.resize(features, expected)
            scores = self.model.predict_proba(
                self._legacy_scaler.transform([features])
            )[0]
            probabilities = {
                self.idx_to_class[i]: float(scores[i]) for i in range(len(scores))
            }
            predicted_idx = int(np.argmax(scores))
            predicted_class = self.idx_to_class[predicted_idx]
            info = CLASS_INFO.get(predicted_class, {})
            return {
                "predicted_class": predicted_class,
                "predicted_label": info.get("label", predicted_class.title()),
                "confidence": float(scores[predicted_idx]),
                "probabilities": probabilities,
                "risk": info.get("risk", "unknown"),
                "note": info.get("note", ""),
            }

        import tensorflow as tf

        raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = self._preprocess_input(img)
        img = tf.expand_dims(img, axis=0)

        predictions = self.model.predict(img, verbose=0)[0]

        probabilities = {
            self.idx_to_class[i]: float(predictions[i])
            for i in range(len(predictions))
        }

        predicted_idx = int(np.argmax(predictions))
        predicted_class = self.idx_to_class[predicted_idx]
        confidence = float(predictions[predicted_idx])
        info = CLASS_INFO.get(predicted_class, {})

        return {
            "predicted_class": predicted_class,
            "predicted_label": info.get("label", predicted_class.title()),
            "confidence": confidence,
            "probabilities": probabilities,
            "risk": info.get("risk", "unknown"),
            "note": info.get("note", ""),
        }
