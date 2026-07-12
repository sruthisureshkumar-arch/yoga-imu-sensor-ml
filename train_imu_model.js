// ============================================================================
// train_imu_model.js — Sensor-only (IMU) yoga step classifier
//
// Trains a neural network on ONLY inertial-sensor features (no camera
// landmarks) to classify which step of which pose the wearer is in, across
// the same 74 step-level classes as the main video pipeline.
//
// Mirrors the evaluation conventions of the reference paper (Patil et al.,
// Sensors 2021, 21, 2340): a fixed calibration/standardization transform,
// per-use-case evaluation, and results reported at two granularities —
// step-level (74 classes) and pose-level (13 classes) — the pose-level
// figure being the direct analog of the paper's coarser "key pose" accuracy.
//
// Algorithm (paper pseudocode style):
//   Inputs:  imu_training_data.json (F ∈ ℝ⁸ per sample), imu_labels.json,
//            imu_norm_params.json {μᵢ, σᵢ}
//   Output:  tfjs_imu_model/, imu_model_config.json (metrics + norm params)
//   Step-1: standardize every sample: zᵢ = (xᵢ − μᵢ) / σᵢ;
//   Step-2: stratified split — per class, 80% train / 20% validation;
//   Step-3: while (epoch < E) do
//             forward pass through MLP  8 → 256 → 128 → 74 (softmax);
//             minimize categorical cross-entropy (Adam, lr = 1e-3);
//             where dropout 0.3/0.2 regularizes between layers;
//           end
//   Step-4: evaluate on validation set:
//             step-level accuracy, macro precision / recall / F1;
//             pose-level accuracy (argmax label mapped to its parent pose);
//
// Usage:  node train_imu_model.js            (run extract_imu_dataset.mjs first)
// ============================================================================

const { tf, saveModel } = require('./tf_env.js');
const fs = require('fs');
const { join } = require('path');

const DATA_FILE = join(__dirname, 'imu_training_data.json');
const NORM_FILE = join(__dirname, 'imu_norm_params.json');
const LABELS_FILE = join(__dirname, 'imu_labels.json');
const MODEL_DIR = join(__dirname, 'tfjs_imu_model');
const CONFIG_FILE = join(__dirname, 'imu_model_config.json');

const EPOCHS = parseInt(process.env.EPOCHS || '40', 10);
const BATCH_SIZE = 256;
const VAL_FRACTION = 0.2;

function poseOfLabel(label) {
  return label.replace(/_Step\d+$/, '');
}

async function main() {
  if (!fs.existsSync(DATA_FILE)) {
    console.error('❌ imu_training_data.json not found — run: node extract_imu_dataset.mjs');
    process.exit(1);
  }

  const samples = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  const labels = JSON.parse(fs.readFileSync(LABELS_FILE, 'utf8'));
  const { mean, std, featureNames } = JSON.parse(fs.readFileSync(NORM_FILE, 'utf8'));
  const numClasses = labels.length;
  const featureSize = featureNames.length;
  const labelIdx = new Map(labels.map((l, i) => [l, i]));

  console.log(`🧘 IMU-only training: ${samples.length} samples, ${featureSize} features, ${numClasses} classes`);

  // Step-1: standardize
  for (const s of samples) {
    s.z = s.features.map((v, i) => (v - mean[i]) / std[i]);
    s.y = labelIdx.get(s.label);
  }

  // Step-2: stratified 80/20 split (deterministic shuffle per class)
  let seed = 42;
  const rand = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
  const byClass = new Map();
  for (const s of samples) {
    if (!byClass.has(s.y)) byClass.set(s.y, []);
    byClass.get(s.y).push(s);
  }
  const train = [], val = [];
  for (const arr of byClass.values()) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    const nVal = Math.max(1, Math.round(arr.length * VAL_FRACTION));
    val.push(...arr.slice(0, nVal));
    train.push(...arr.slice(nVal));
  }
  // shuffle train
  for (let i = train.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [train[i], train[j]] = [train[j], train[i]];
  }
  console.log(`   train ${train.length} / val ${val.length} (stratified)`);

  const xTrain = tf.tensor2d(train.map(s => s.z));
  const yTrain = tf.oneHot(train.map(s => s.y), numClasses);
  const xVal = tf.tensor2d(val.map(s => s.z));
  const yVal = tf.oneHot(val.map(s => s.y), numClasses);

  // Step-3: MLP
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [featureSize], units: 256, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(1e-3), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  model.summary();

  await model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [xVal, yVal],
    verbose: 0,
    callbacks: {
      onEpochEnd: (e, logs) => {
        if ((e + 1) % 5 === 0 || e === 0) {
          console.log(`   epoch ${String(e + 1).padStart(3)}: loss ${logs.loss.toFixed(4)}  acc ${(logs.acc * 100).toFixed(1)}%  val_acc ${(logs.val_acc * 100).toFixed(1)}%`);
        }
      },
    },
  });

  // Step-4: evaluation
  const predT = model.predict(xVal);
  const predIdx = (await predT.argMax(-1).data());
  predT.dispose();

  const conf = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  let stepCorrect = 0, poseCorrect = 0;
  for (let i = 0; i < val.length; i++) {
    const t = val[i].y, p = predIdx[i];
    conf[t][p]++;
    if (t === p) stepCorrect++;
    if (poseOfLabel(labels[t]) === poseOfLabel(labels[p])) poseCorrect++;
  }
  const stepAcc = stepCorrect / val.length;
  const poseAcc = poseCorrect / val.length;

  // macro precision / recall / F1 (step level)
  let mp = 0, mr = 0, mf = 0, counted = 0;
  for (let c = 0; c < numClasses; c++) {
    const tp = conf[c][c];
    const fn = conf[c].reduce((a, b) => a + b, 0) - tp;
    let fp = 0;
    for (let r = 0; r < numClasses; r++) if (r !== c) fp += conf[r][c];
    if (tp + fn === 0) continue;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp / (tp + fn);
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    mp += prec; mr += rec; mf += f1; counted++;
  }
  mp /= counted; mr /= counted; mf /= counted;

  console.log('\n📊 Validation results (sensor-only input):');
  console.log(`   Step-level  (${numClasses} classes): accuracy ${(stepAcc * 100).toFixed(2)}%`);
  console.log(`   Pose-level  (13 poses)             : accuracy ${(poseAcc * 100).toFixed(2)}%`);
  console.log(`   Macro precision ${(mp * 100).toFixed(2)}%  recall ${(mr * 100).toFixed(2)}%  F1 ${(mf * 100).toFixed(2)}%`);

  await saveModel(model, MODEL_DIR);
  fs.writeFileSync(CONFIG_FILE, JSON.stringify({
    trainedAt: new Date().toISOString(),
    inputModality: 'imu-only',
    featureNames,
    featureSize,
    numClasses,
    epochs: EPOCHS,
    trainSamples: train.length,
    valSamples: val.length,
    normalization: { mean, std },
    metrics: {
      stepAccuracy: +stepAcc.toFixed(4),
      poseAccuracy: +poseAcc.toFixed(4),
      macroPrecision: +mp.toFixed(4),
      macroRecall: +mr.toFixed(4),
      macroF1: +mf.toFixed(4),
    },
  }, null, 2));

  console.log(`\n✅ Saved model → ${MODEL_DIR}`);
  console.log(`✅ Saved config + metrics → ${CONFIG_FILE}`);

  tf.dispose([xTrain, yTrain, xVal, yVal]);
}

main().catch(e => { console.error('❌', e); process.exit(1); });
