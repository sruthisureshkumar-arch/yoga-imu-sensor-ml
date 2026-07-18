# yoga-imu-sensor-ml — Sensor-Only Yoga Step Classification

Standalone companion project to **yoga-pose-correction-ai**. Trains a machine-learning
model on **only the 6-axis IMU sensor data** (no camera landmarks) to classify the 74
pose-step classes of the Common Yoga Protocol dataset.

Modeled on the multi-sensor architecture of *Patil et al., "An Open-Source Platform for
Human Pose Estimation and Tracking Using a Heterogeneous Multi-Sensor System," Sensors
2021, 21, 2340* — this is the inertial-only branch of that design, applied to our own
recordings. See `System_Workflow_IMU_Sensor_Pipeline.md` for the full specification in
the paper's algorithm format.

## Data

Sources sensor packets from the parent project's `training_data.json` (read-only — the
parent project is never modified). Of 164,670 samples, the 29,116 from the original
7-participant batch carry non-zero accelerometer/gyroscope values and are used here.

> **⚠️ Important:** a statistical audit shows these non-zero values are **synthetic
> uniform-noise placeholders, not real sensor recordings** (see Section 4 of the
> workflow doc for the evidence). Current model metrics are therefore chance-level by
> construction. The pipeline is verified end-to-end and will train meaningfully as
> soon as real IMU recordings are merged into the dataset — no code changes needed.

| | |
|---|---|
| Input | `[ax, ay, az, gx, gy, gz]` + derived `‖a‖`, `‖ω‖` → 8 features |
| Classes | 74 pose-steps (13 poses) — same label set as the video pipeline |
| Model | MLP 8 → 256 → 128 → 74, dropout 0.3/0.2, Adam |
| Split | Stratified 80/20 |
| Decision | Mamdani fuzzy inference (motion intensity × classifier confidence → graded step-hold degree) |

## Usage

```bash
npm install            # tfjs-node (falls back to pure-JS tfjs if native build unavailable)
npm run extract        # build imu_training_data.json from ../training_data.json
npm run train          # train + evaluate → tfjs_imu_model/, imu_model_config.json
npm run predict        # top-3 prediction + fuzzy hold-degree demo
node fuzzy_decision.mjs             # fuzzy layer standalone demo cases
node fuzzy_decision.mjs 0.5 0.1 0.85   # ‖a‖ ‖ω‖ confidence → holdDegree
```

To point the extractor at a dataset elsewhere:
`node extract_imu_dataset.mjs /path/to/training_data.json`

## Outputs

- `tfjs_imu_model/` — trained TensorFlow.js model
- `imu_model_config.json` — feature spec, normalization (μ/σ), and validation metrics
  (step-level accuracy, pose-level accuracy, macro precision/recall/F1)
- `imu_labels.json` — class index → step label

## Relationship to the main project

Independent by design: separate package, separate model, nothing in the parent folder is
touched. The feature construction and normalization here are built so the trained model
can later be fused back into the parent pipeline's reserved IMU feature slots
(camera + IMU fusion — the full analog of the reference paper's lidar + IMU system).
