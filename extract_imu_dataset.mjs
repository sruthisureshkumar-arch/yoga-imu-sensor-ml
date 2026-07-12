// ============================================================================
// extract_imu_dataset.mjs — Sensor-only dataset builder
//
// Pulls ONLY the inertial-sensor (IMU) channels out of the main project's
// training_data.json, keeping just the samples that carry real (non-zero)
// accelerometer/gyroscope readings — i.e. the original 7-participant batch
// recorded with synchronized 6-axis IMU packets.
//
// Follows the data-preparation conventions of the reference paper:
//   Patil et al., "An Open-Source Platform for Human Pose Estimation and
//   Tracking Using a Heterogeneous Multi-Sensor System", Sensors 2021, 21, 2340.
// The paper fuses lidar + 10 IMUs; this pipeline is the IMU-only branch of
// that idea, applied to our own recordings: raw inertial channels are taken
// as the sole input modality (no camera landmarks), and per-channel
// standardization parameters are computed here (the "calibration" step) and
// saved so training and live inference share one fixed transform.
//
// Algorithm (paper pseudocode style):
//   Inputs:  training_data.json samples { label, imuFeatures = [ax,ay,az,gx,gy,gz] }
//   Output:  imu_training_data.json, imu_norm_params.json, imu_labels.json
//   for each sample s do
//     Step-1: if all IMU channels are zero → discard (no real sensor packet);
//     Step-2: F_raw = [ax, ay, az, gx, gy, gz];
//     Step-3: derive magnitudes: |a| = √(ax²+ay²+az²), |ω| = √(gx²+gy²+gz²);
//             F = F_raw ⊕ [|a|, |ω|],  F ∈ ℝ⁸;
//   end
//   Step-4: per-channel mean μᵢ and std σᵢ over all kept samples;
//           saved to imu_norm_params.json (applied as zᵢ = (xᵢ − μᵢ)/σᵢ at
//           train + inference time).
//
// Usage:  node extract_imu_dataset.mjs [path/to/training_data.json]
//         (default: ../training_data.json — the main project's dataset)
// ============================================================================

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const SOURCE = process.argv[2] || join(__dirname, '..', 'training_data.json');
const OUT_DATA = join(__dirname, 'imu_training_data.json');
const OUT_NORM = join(__dirname, 'imu_norm_params.json');
const OUT_LABELS = join(__dirname, 'imu_labels.json');

export const IMU_CHANNEL_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz'];
export const FEATURE_NAMES = [...IMU_CHANNEL_NAMES, 'accelMagnitude', 'gyroMagnitude'];
export const FEATURE_SIZE = FEATURE_NAMES.length; // 8

// Step-2/3: raw 6-axis channels + derived magnitudes → F ∈ ℝ⁸
export function buildImuFeatureVector(imu6) {
  const [ax, ay, az, gx, gy, gz] = imu6;
  const accelMag = Math.sqrt(ax * ax + ay * ay + az * az);
  const gyroMag = Math.sqrt(gx * gx + gy * gy + gz * gz);
  return [ax, ay, az, gx, gy, gz, accelMag, gyroMag];
}

function main() {
  console.log(`📥 Reading source dataset: ${SOURCE}`);
  const raw = JSON.parse(fs.readFileSync(SOURCE, 'utf8'));
  console.log(`   ${raw.length} total samples`);

  // Step-1: keep only samples with a real (non-zero) IMU packet
  const kept = [];
  for (const s of raw) {
    const imu = s.imuFeatures;
    if (!Array.isArray(imu) || imu.length !== 6) continue;
    if (imu.every(v => v === 0)) continue;
    kept.push({
      label: s.label,
      poseName: s.poseName,
      stepIndex: s.stepIndex,
      features: buildImuFeatureVector(imu),
    });
  }
  console.log(`   ${kept.length} samples carry real 6-axis IMU data (${(100 * kept.length / raw.length).toFixed(1)}%)`);

  // Label index (sorted for determinism)
  const labels = [...new Set(kept.map(s => s.label))].sort();
  console.log(`   ${labels.length} step-level classes`);

  // Step-4: per-channel standardization parameters ("calibration")
  const n = kept.length;
  const mean = new Array(FEATURE_SIZE).fill(0);
  for (const s of kept) for (let i = 0; i < FEATURE_SIZE; i++) mean[i] += s.features[i];
  for (let i = 0; i < FEATURE_SIZE; i++) mean[i] /= n;

  const std = new Array(FEATURE_SIZE).fill(0);
  for (const s of kept) for (let i = 0; i < FEATURE_SIZE; i++) std[i] += (s.features[i] - mean[i]) ** 2;
  for (let i = 0; i < FEATURE_SIZE; i++) std[i] = Math.sqrt(std[i] / n) || 1;

  fs.writeFileSync(OUT_DATA, JSON.stringify(kept));
  fs.writeFileSync(OUT_NORM, JSON.stringify({ featureNames: FEATURE_NAMES, mean, std }, null, 2));
  fs.writeFileSync(OUT_LABELS, JSON.stringify(labels, null, 2));

  console.log(`\n✅ Wrote:`);
  console.log(`   ${OUT_DATA}  (${n} samples × ${FEATURE_SIZE} features)`);
  console.log(`   ${OUT_NORM}  (per-channel μ/σ)`);
  console.log(`   ${OUT_LABELS} (${labels.length} classes)`);
}

// Run only when executed directly (this module is also imported by predict_imu.mjs)
if (process.argv[1] === fileURLToPath(import.meta.url)) main();
