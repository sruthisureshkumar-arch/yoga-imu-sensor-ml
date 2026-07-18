// ============================================================================
// predict_imu.mjs — Sensor-only inference demo
//
// Feeds a single 6-axis IMU packet [ax, ay, az, gx, gy, gz] through the
// trained IMU-only classifier and prints the top-3 predicted pose steps.
// The same standardization saved at extraction time (imu_model_config.json)
// is applied, so inference matches the training distribution exactly.
//
// Usage:
//   node predict_imu.mjs 1.91 -0.95 -1.45 0.37 -0.44 0.85
//   node predict_imu.mjs            (uses a random validation sample)
// ============================================================================

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { buildImuFeatureVector } from './extract_imu_dataset.mjs';
import { createFuzzySystem } from './fuzzy_decision.mjs';

const require = createRequire(import.meta.url);
const { tf, loadModel } = require('./tf_env.js');

const __dirname = dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(fs.readFileSync(join(__dirname, 'imu_model_config.json'), 'utf8'));
const labels = JSON.parse(fs.readFileSync(join(__dirname, 'imu_labels.json'), 'utf8'));

async function main() {
  let imu6;
  if (process.argv.length >= 8) {
    imu6 = process.argv.slice(2, 8).map(Number);
  } else {
    const data = JSON.parse(fs.readFileSync(join(__dirname, 'imu_training_data.json'), 'utf8'));
    const s = data[Math.floor(Math.random() * data.length)];
    imu6 = s.features.slice(0, 6);
    console.log(`(random sample — true label: ${s.label})`);
  }

  const feats = buildImuFeatureVector(imu6);
  const { mean, std } = config.normalization;
  const z = feats.map((v, i) => (v - mean[i]) / std[i]);

  const model = await loadModel(join(__dirname, 'tfjs_imu_model'));
  const scores = await model.predict(tf.tensor2d([z])).data();

  const top = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, 3);
  console.log('\nIMU packet:', imu6.join(', '));
  console.log('Top-3 predictions:');
  for (const i of top) console.log(`  ${(scores[i] * 100).toFixed(1).padStart(5)}%  ${labels[i]}`);

  // Fuzzy decision layer: motion intensity (‖a‖, ‖ω‖) + classifier
  // confidence → graded, interpretable step-hold degree.
  const fis = createFuzzySystem();
  const [accelMag, gyroMag] = [feats[6], feats[7]];
  const fz = fis.evaluate(accelMag, gyroMag, scores[top[0]]);
  console.log(`\nFuzzy decision: holdDegree ${fz.holdDegree} → ${fz.label}`);
  console.log(`  motion score ${fz.motionScore}  |  fired rules: ${fz.firedRules.map(f => `${f.rule}→${f.output}@${f.strength}`).join(', ') || 'none'}`);
}

main().catch(e => { console.error('❌', e); process.exit(1); });
