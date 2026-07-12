# System Workflow — Sensor-Only (IMU) Yoga Step Classification Pipeline

*Companion module to the video-based pose correction pipeline (yoga-pose-correction-ai).*

**Reference paper:** A. K. Patil, A. Balasubramanyam, J. Y. Ryu, B. Chakravarthi, and Y. H. Chai,
"An Open-Source Platform for Human Pose Estimation and Tracking Using a Heterogeneous
Multi-Sensor System," *Sensors* 2021, 21, 2340. https://doi.org/10.3390/s21072340

## 1. Overview

The reference paper estimates human pose by fusing a heterogeneous pair of sensor
modalities — lidar (position) and 10 body-worn IMUs (orientation) — and reports that a
single inertial modality alone accumulates drift and is therefore usually combined with a
second modality. This module implements the **inertial-only branch** of that architecture
for our own dataset: it takes ONLY the 6-axis IMU packets recorded synchronously with our
yoga sessions (no camera landmarks) and trains a neural classifier to identify which step
of which Common Yoga Protocol pose the wearer is in, across the same 74 step-level classes
used by the video pipeline.

As in the paper, algorithms below are written in the Inputs / Output / while-loop
pseudocode style, with formulas inline and variable definitions on "where" lines.

## 2. System Workflow Diagram

```
6-axis IMU packets ──► Packet filtering ──► Feature construction ──► Standardization
[ax ay az gx gy gz]     (discard all-zero)    F ∈ ℝ⁸ (+‖a‖, ‖ω‖)      zᵢ = (xᵢ−μᵢ)/σᵢ
                                                                          │
                    Step + pose prediction ◄── MLP classifier ◄───────────┘
                    (74 / 13 classes)           8→256→128→74
```

*Figure 1. Proposed system workflow (sensor-only pipeline; camera/landmark module excluded).*

## 3. Workflow Modules

### 3.1 Sensor Data Acquisition

Each training sample in the parent dataset (`training_data.json`, 164,670 samples) carries
a synchronized 6-value IMU vector `[ax, ay, az, gx, gy, gz]` — tri-axial accelerometer and
gyroscope — recorded alongside the MediaPipe landmarks. Only the original 7-participant
batch (29,116 samples, 17.7%) contains real, non-zero inertial readings; later batches were
recorded with the 10-sensor BNO085 array whose firmware currently streams zero accel/gyro,
and are excluded. This mirrors the paper's Xsens MTW setup in role (body-worn inertial
sensing at fixed rate), reduced to a single 6-axis unit.

### 3.2 Feature Construction

**Algorithm 1: IMU feature extraction and calibration.**

```
Inputs:
  • Raw IMU packets P = { [ax, ay, az, gx, gy, gz] } with labels;
Output:
  • Feature matrix F ∈ ℝ^(N×8), calibration parameters {μᵢ, σᵢ};

for each packet p in P do
  Step-1: if all channels of p are zero → discard;
          (no real sensor payload — firmware placeholder packet)
  Step-2: F_raw = [ax, ay, az, gx, gy, gz];
  Step-3: ‖a‖ = √(ax² + ay² + az²);  ‖ω‖ = √(gx² + gy² + gz²);
          F = F_raw ⊕ [‖a‖, ‖ω‖],  |F| = 8;
          where ‖a‖ is total specific force and ‖ω‖ total angular rate —
          orientation-independent motion-intensity descriptors;
end
Step-4: μᵢ = mean(Fᵢ), σᵢ = std(Fᵢ) over all kept samples, i = 1..8;
        saved once (calibration) and reused identically at inference,
        analogous to the paper's one-time attention-pose calibration.
```

### 3.3 Step Classification

**Algorithm 2: Sensor-only step classification.**

```
Inputs:
  • Standardized features z, zᵢ = (xᵢ − μᵢ)/σᵢ;
  • Class labels y ∈ {0..73} (74 pose-step classes);
Output:
  • Trained classifier; step-level and pose-level predictions;

Step-1: Stratified split — per class, 80% train / 20% validation;
Step-2: while (epoch < E) do
          ŷ = softmax(W₃·ReLU(W₂·ReLU(W₁·z)));
          minimize L = −Σ y·log(ŷ)  (Adam, α = 10⁻³);
          where W₁ ∈ ℝ^(256×8), W₂ ∈ ℝ^(128×256), W₃ ∈ ℝ^(74×128),
          with dropout 0.3 / 0.2 between layers;
        end
Step-3: step-level prediction: ĉ = argmax(ŷ);
        pose-level prediction: pose(ĉ), stripping the _StepN suffix;
```

### 3.4 Evaluation Protocol

Following the paper's two-granularity reporting (fine joint-level vs. coarse key-pose
accuracy), results are reported at:

1. **Step level** — 74 classes, accuracy + macro precision / recall / F1;
2. **Pose level** — 13 poses, accuracy of the parent pose of the predicted step.

Metrics are written to `imu_model_config.json` at the end of every training run.

## 4. Results

See `imu_model_config.json` → `metrics` for the current trained model's figures
(fills automatically on each `npm run train`).

**⚠️ Data audit finding (July 2026).** A statistical audit of the 29,116 "non-zero"
IMU samples shows they are **synthetic placeholder values, not real sensor
recordings**: every channel matches a uniform random distribution exactly (excess
kurtosis ≈ −1.20 on all six channels; uniform = −1.20, real IMU data is heavy-tailed),
and per-class accelerometer means are ≈ 0 for standing, prone, and supine poses
alike — physically impossible for a real accelerometer, where the ~1 g gravity
component would appear on different axes for a standing versus lying participant.
Consequently the classifier converges to chance-level accuracy (~1.4% over 74
classes), which is the mathematically expected outcome of training on noise, and the
metrics in `imu_model_config.json` should be read as a **pipeline verification run,
not a performance result**. The full pipeline (extraction → calibration → training →
two-granularity evaluation) is verified end-to-end and will produce meaningful
accuracy the moment genuine sensor recordings replace the placeholders — no code
changes needed, just re-run `npm run extract && npm run train` against the updated
dataset.

Once real data is present: consistent with the reference paper's motivation — a
single inertial modality is insufficient for precise pose estimation and benefits
from fusion with a positional modality — the sensor-only classifier is expected to
trail the 111-feature video pipeline (89.1% step-level validation accuracy). Its
value is as: (a) the standalone IMU baseline quantifying how much signal the wearable
channel carries, and (b) the building block for the future camera + IMU fusion model,
in which these features re-enter the parent pipeline's reserved IMU slots.

## 5. Future Work

- Replace the single 6-axis input with the finalized 10-sensor BNO085 array (one per
  major limb segment plus upper/lower spine), each streaming accel + gyro + quaternion,
  once firmware streams non-zero inertial values — bringing the setup to full parity
  with the paper's 10-IMU configuration.
- Temporal windows (the paper operates on 60 Hz streams, not single packets): stack k
  consecutive packets or add an LSTM front-end to capture motion dynamics between steps.
- Heterogeneous fusion: merge this classifier's features with the video pipeline's
  landmark features, replicating the paper's lidar+IMU fusion with camera+IMU.
