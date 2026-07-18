// ============================================================================
// fuzzy_decision.mjs — Mamdani fuzzy inference decision layer
//
// Sits ON TOP of the neural step classifier. Instead of a crisp
// "match / no-match" cut on the softmax score, it combines two graded inputs:
//
//   1. Motion intensity — how much the wearer is currently moving, from the
//      orientation-independent magnitudes ‖a‖ (specific force) and ‖ω‖
//      (angular rate). A yoga step is a HELD position: high angular rate
//      means the wearer is transitioning between steps, so even a confident
//      classification shouldn't count as "holding the step".
//   2. Classifier confidence — the top softmax probability from the MLP.
//
// and outputs a defuzzified "step-hold degree" ∈ [0, 1] with a linguistic
// label (Transitioning / Adjusting / Holding) and the list of fired rules,
// keeping the decision fully interpretable.
//
// Membership functions for motion are DATA-DRIVEN: parameterized from the
// dataset's own μ/σ of ‖a‖ and ‖ω‖ (imu_norm_params.json) — the fuzzy
// analog of the pipeline's one-time calibration step.
//
// Fuzzy sets:
//   motion      : Still, Slow, Fast            (from ‖ω‖, ‖a‖ vs. calibration)
//   confidence  : Low, Medium, High            (fixed sets on [0, 1])
//   holdDegree  : Low, Medium, High            (output sets on [0, 1])
//
// Rule base (Mamdani, max-min composition, centroid defuzzification):
//   R1: IF motion is Still AND confidence is High   THEN holdDegree is High
//   R2: IF motion is Still AND confidence is Medium THEN holdDegree is Medium
//   R3: IF motion is Slow  AND confidence is High   THEN holdDegree is Medium
//   R4: IF motion is Fast                           THEN holdDegree is Low
//   R5: IF confidence is Low                        THEN holdDegree is Low
// ============================================================================

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/* ── Membership function primitives ────────────────────────────────────── */

export const trimf = (x, [a, b, c]) =>
  x <= a || x >= c ? 0 : x <= b ? (x - a) / (b - a || 1) : (c - x) / (c - b || 1);

export const trapmf = (x, [a, b, c, d]) =>
  x <= a || x >= d ? 0
  : x < b ? (x - a) / (b - a || 1)
  : x <= c ? 1
  : (d - x) / (d - c || 1);

/* ── Fuzzy system construction ─────────────────────────────────────────── */

/**
 * Builds the fuzzy inference system. Motion membership functions are
 * calibrated from the dataset statistics (μ, σ of ‖a‖ and ‖ω‖); pass no
 * argument to load imu_norm_params.json from this folder.
 */
export function createFuzzySystem(normParams) {
  if (!normParams) {
    normParams = JSON.parse(fs.readFileSync(join(__dirname, 'imu_norm_params.json'), 'utf8'));
  }
  const iA = normParams.featureNames.indexOf('accelMagnitude');
  const iG = normParams.featureNames.indexOf('gyroMagnitude');
  const [muA, sdA] = [normParams.mean[iA], normParams.std[iA]];
  const [muG, sdG] = [normParams.mean[iG], normParams.std[iG]];

  // Motion intensity x: weighted blend of standardized magnitudes; gyro is
  // weighted higher — angular rate is the clearest "in transition" signal.
  const motionScore = (accelMag, gyroMag) => {
    const zA = (accelMag - muA) / sdA;
    const zG = (gyroMag - muG) / sdG;
    return 0.35 * zA + 0.65 * zG; // typically ∈ [-3, 3]
  };

  const motionSets = {
    Still: x => trapmf(x, [-10, -9, -0.8, 0.0]),
    Slow:  x => trimf(x, [-0.8, 0.0, 0.8]),
    Fast:  x => trapmf(x, [0.0, 0.8, 9, 10]),
  };

  const confidenceSets = {
    Low:    p => trapmf(p, [-1, 0, 0.20, 0.40]),
    Medium: p => trimf(p, [0.20, 0.45, 0.70]),
    High:   p => trapmf(p, [0.50, 0.75, 1.0, 2]),
  };

  // Output sets over holdDegree ∈ [0, 1]
  const outputSets = {
    Low:    y => trapmf(y, [-1, 0, 0.15, 0.40]),
    Medium: y => trimf(y, [0.25, 0.50, 0.75]),
    High:   y => trapmf(y, [0.60, 0.85, 1.0, 2]),
  };

  const rules = [
    { name: 'R1', out: 'High',   fire: m => Math.min(m.motion.Still, m.conf.High) },
    { name: 'R2', out: 'Medium', fire: m => Math.min(m.motion.Still, m.conf.Medium) },
    { name: 'R3', out: 'Medium', fire: m => Math.min(m.motion.Slow, m.conf.High) },
    { name: 'R4', out: 'Low',    fire: m => m.motion.Fast },
    { name: 'R5', out: 'Low',    fire: m => m.conf.Low },
  ];

  /**
   * Evaluate the FIS.
   * @param accelMag  raw ‖a‖ of the current packet
   * @param gyroMag   raw ‖ω‖ of the current packet
   * @param confidence top softmax probability from the classifier
   * @returns { holdDegree, label, motionScore, memberships, firedRules }
   */
  function evaluate(accelMag, gyroMag, confidence) {
    const x = motionScore(accelMag, gyroMag);

    const memberships = {
      motion: Object.fromEntries(Object.entries(motionSets).map(([k, f]) => [k, f(x)])),
      conf: Object.fromEntries(Object.entries(confidenceSets).map(([k, f]) => [k, f(confidence)])),
    };

    // Mamdani: rule strength clips its output set; aggregate with max
    const strengths = { Low: 0, Medium: 0, High: 0 };
    const firedRules = [];
    for (const r of rules) {
      const s = r.fire(memberships);
      if (s > 0) firedRules.push({ rule: r.name, output: r.out, strength: +s.toFixed(3) });
      strengths[r.out] = Math.max(strengths[r.out], s);
    }

    // Centroid defuzzification (discretized [0, 1])
    let num = 0, den = 0;
    for (let y = 0; y <= 1.0001; y += 0.01) {
      const mu = Math.max(
        Math.min(strengths.Low, outputSets.Low(y)),
        Math.min(strengths.Medium, outputSets.Medium(y)),
        Math.min(strengths.High, outputSets.High(y)),
      );
      num += y * mu;
      den += mu;
    }
    const holdDegree = den > 0 ? num / den : 0;

    const label = holdDegree >= 0.66 ? 'Holding'
      : holdDegree >= 0.40 ? 'Adjusting'
      : 'Transitioning';

    return {
      holdDegree: +holdDegree.toFixed(4),
      label,
      motionScore: +x.toFixed(4),
      memberships,
      firedRules,
    };
  }

  return { evaluate, motionScore };
}

/* ── CLI demo:  node fuzzy_decision.mjs <accelMag> <gyroMag> <confidence> ── */
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [aMag, gMag, conf] = process.argv.slice(2, 5).map(Number);
  const fis = createFuzzySystem();
  const cases = Number.isFinite(conf)
    ? [[aMag, gMag, conf]]
    : [[0.5, 0.1, 0.85], [2.0, 0.9, 0.85], [1.0, 0.5, 0.45], [3.0, 1.5, 0.10]];
  for (const [a, g, c] of cases) {
    const r = fis.evaluate(a, g, c);
    console.log(`‖a‖=${a} ‖ω‖=${g} conf=${c} → holdDegree ${r.holdDegree} (${r.label})  rules: ${r.firedRules.map(f => `${f.rule}→${f.output}@${f.strength}`).join(', ') || 'none'}`);
  }
}
