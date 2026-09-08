# Bundled artifacts

The three original model files were moved here without modifying their bytes:

| Current file | Original file |
| --- | --- |
| `exercise_bilstm.h5` | `final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5` |
| `feature_scaler.pkl` | `thesis_bidirectionallstm_scaler.pkl` |
| `label_encoder.pkl` | `thesis_bidirectionallstm_label_encoder.pkl` |

`manifest.json` records SHA-256 checksums and metadata read from the artifacts.
The HDF5 file declares Keras 3.3.3; both sklearn objects declare version 1.5.0.
The model contains two bidirectional LSTM layers (91 units in each direction),
dropout after each, and a four-class softmax Dense output. Its input is `(N,30,22)`;
the scaler accepts flattened `(N,660)` windows.

Training videos, training code, hyperparameter search, split membership, subject
counts and measured validation results were not supplied. The empty training
scripts have been removed. This project preserves inference; it cannot reproduce
the original training or establish that new test subjects were absent from it.

Load only these trusted files. Joblib/pickle can execute code. Uploading or
replacing models through the frontend is intentionally unsupported.
