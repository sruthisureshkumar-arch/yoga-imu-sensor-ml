// ============================================================================
// tf_env.js — TensorFlow environment shim
//
// Prefers the native @tensorflow/tfjs-node backend (fast, used on macOS/
// normal installs). Falls back to the pure-JS @tensorflow/tfjs backend when
// the native binding isn't available, with manual file save/load handlers
// (pure-JS tfjs has no file:// IO in Node).
// ============================================================================

const fs = require('fs');
const { join } = require('path');

let tf, native = true;
try {
  tf = require('@tensorflow/tfjs-node');
} catch {
  tf = require('@tensorflow/tfjs');
  native = false;
  console.log('ℹ️  Using pure-JS TensorFlow backend (tfjs-node native binding unavailable)');
}

async function saveModel(model, dir) {
  if (native) {
    await model.save(`file://${dir}`);
    return;
  }
  fs.mkdirSync(dir, { recursive: true });
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightData = artifacts.weightData instanceof ArrayBuffer
      ? artifacts.weightData
      : artifacts.weightData.buffer;
    fs.writeFileSync(join(dir, 'weights.bin'), Buffer.from(weightData));
    fs.writeFileSync(join(dir, 'model.json'), JSON.stringify({
      modelTopology: artifacts.modelTopology,
      format: 'layers-model',
      generatedBy: 'yoga-imu-sensor-ml',
      convertedBy: null,
      weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
    }));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));
}

async function loadModel(dir) {
  if (native) {
    return tf.loadLayersModel(`file://${join(dir, 'model.json')}`);
  }
  const spec = JSON.parse(fs.readFileSync(join(dir, 'model.json'), 'utf8'));
  const weightData = fs.readFileSync(join(dir, spec.weightsManifest[0].paths[0]));
  return tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: spec.modelTopology,
    weightSpecs: spec.weightsManifest[0].weights,
    weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
  }));
}

module.exports = { tf, native, saveModel, loadModel };
