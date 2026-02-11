import java.io.*;
import java.nio.*;
import java.util.*;

/**
 * SynapticAI Training Pipeline - Self-contained Java implementation.
 *
 * Trains a Sparse Mixture-of-Sensors model on synthetic sensor data,
 * then exports weights for the Android app.
 *
 * Architecture trained:
 * - 6 sensor expert projections (Motion, Spatial, Visual, Audio, Ambient, Radio)
 * - Expert routing gate (learns which sensors matter per context)
 * - Cross-sensor attention (learns sensor correlations)
 * - Feed-forward network
 * - Context classification head (16 activities)
 * - Anomaly detection head
 */
public class TrainSynaptic {

    // Model dimensions
    static final int[] SENSOR_DIMS = {9, 7, 64, 40, 4, 8}; // Per-modality raw dimensions
    static final int TOTAL_SENSOR_DIM = 132;
    static final int EXPERT_DIM = 32;
    static final int FUSED_DIM = 192;  // 6 * 32
    static final int ATTN_DIM = 64;
    static final int FFN_DIM = 128;
    static final int NUM_CONTEXTS = 16;
    static final int NUM_EXPERTS = 6;

    static final String[] CONTEXT_NAMES = {
        "WALKING", "SITTING_DESK", "DRIVING", "IN_MEETING", "SLEEPING",
        "COOKING", "EXERCISING", "ELEVATOR", "PHONE_CALL", "OUTDOORS_NATURE",
        "PUBLIC_TRANSIT", "SHOPPING", "CLIMBING_STAIRS", "CYCLING", "IDLE_TABLE", "TRANSITION"
    };

    static final String[] EXPERT_NAMES = {"MOTION", "SPATIAL", "VISUAL", "AUDIO", "AMBIENT", "RADIO"};

    // ===================== MODEL PARAMETERS =====================

    static float[][] sensorProjections = new float[6][];
    static float[][] sensorBiases = new float[6][];
    static float[] routingGate, routingBias;
    static float[] crossAttnQ, crossAttnK, crossAttnV, crossAttnOut;
    static float[] ffnW1, ffnB1, ffnW2, ffnB2;
    static float[] contextHead, contextBias;
    static float[] anomalyHead, anomalyBias;

    // AdamW state
    static List<float[]> allParams;
    static List<float[]> mState, vState;
    static int adamStep = 0;

    static Random rng = new Random(42);

    // ===================== MAIN =====================

    public static void main(String[] args) {
        System.out.println();
        System.out.println("  ╔═══════════════════════════════════════════════════╗");
        System.out.println("  ║          SynapticAI Training Pipeline             ║");
        System.out.println("  ║   Sparse Mixture-of-Sensors Transformer v1.0     ║");
        System.out.println("  ║   Target: Samsung S25 Ultra (Snapdragon 8 Elite) ║");
        System.out.println("  ╚═══════════════════════════════════════════════════╝");
        System.out.println();

        // Initialize model
        initializeModel();
        int totalParams = allParams.stream().mapToInt(a -> a.length).sum();
        System.out.println("Model initialized: " + totalParams + " trainable parameters");

        // Generate synthetic data
        System.out.println("Generating synthetic sensor data for 16 activity contexts...");
        int samplesPerContext = 150;
        List<float[][]> trainSensors = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();
        List<Float> trainAnomalyTargets = new ArrayList<>();

        for (int ctx = 0; ctx < NUM_CONTEXTS; ctx++) {
            for (int i = 0; i < samplesPerContext; i++) {
                float phase = rng.nextFloat();
                float[][] sample = generateSensorSample(ctx, phase);
                trainSensors.add(sample);
                trainLabels.add(ctx == 15 ? rng.nextInt(15) : ctx); // Transition gets random label
                trainAnomalyTargets.add(0.0f);
            }
        }

        // Add anomaly samples
        for (int i = 0; i < samplesPerContext; i++) {
            int ctx = rng.nextInt(15);
            float[][] sample = generateAnomalousSample(ctx);
            trainSensors.add(sample);
            trainLabels.add(ctx);
            trainAnomalyTargets.add(0.5f + rng.nextFloat() * 0.5f);
        }

        // Shuffle
        int n = trainSensors.size();
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            Collections.swap(trainSensors, i, j);
            Collections.swap(trainLabels, i, j);
            Collections.swap(trainAnomalyTargets, i, j);
        }

        // Split train/val
        int valSize = (int)(n * 0.15);
        int trainSize = n - valSize;
        System.out.println("Train: " + trainSize + " samples, Validation: " + valSize + " samples");

        // Pre-training eval
        float preAcc = evaluate(trainSensors, trainLabels, trainSize, n);
        System.out.printf("Pre-training accuracy: %.1f%% (random baseline: ~6%%)%n%n", preAcc * 100);

        // ===================== TRAINING LOOP =====================
        int epochs = 30;
        int batchSize = 16;
        float baseLr = 0.003f;
        float bestValAcc = 0;

        System.out.println("Starting training (" + epochs + " epochs, batch size " + batchSize + ")...");
        System.out.println();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long epochStart = System.currentTimeMillis();

            // Cosine annealing LR with warmup
            float lr;
            int warmup = 3;
            if (epoch <= warmup) {
                lr = baseLr * epoch / warmup;
            } else {
                float progress = (float)(epoch - warmup) / (epochs - warmup);
                lr = baseLr * 0.5f * (1 + (float)Math.cos(Math.PI * progress));
            }

            // Shuffle training portion
            for (int i = trainSize - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                Collections.swap(trainSensors, i, j);
                Collections.swap(trainLabels, i, j);
                Collections.swap(trainAnomalyTargets, i, j);
            }

            float epochLoss = 0;
            int epochSamples = 0;

            for (int bStart = 0; bStart < trainSize; bStart += batchSize) {
                int bEnd = Math.min(bStart + batchSize, trainSize);

                // Accumulate gradients via SPSA
                List<float[]> gradAccum = new ArrayList<>();
                for (float[] p : allParams) gradAccum.add(new float[p.length]);

                for (int bi = bStart; bi < bEnd; bi++) {
                    List<float[]> grads = computeSPSAGradients(
                        trainSensors.get(bi), trainLabels.get(bi), trainAnomalyTargets.get(bi));

                    float batchScale = 1.0f / (bEnd - bStart);
                    for (int p = 0; p < grads.size(); p++) {
                        for (int i = 0; i < grads.get(p).length; i++) {
                            gradAccum.get(p)[i] += grads.get(p)[i] * batchScale;
                        }
                    }

                    // Track loss
                    float[] result = forward(trainSensors.get(bi));
                    epochLoss += computeLoss(result, trainLabels.get(bi), trainAnomalyTargets.get(bi));
                    epochSamples++;
                }

                // Gradient clipping
                clipGradients(gradAccum, 1.0f);

                // AdamW step
                adamWStep(gradAccum, lr);
            }

            float avgLoss = epochLoss / epochSamples;

            // Evaluate
            float trainAcc = evaluate(trainSensors, trainLabels, 0, Math.min(300, trainSize));
            float valAcc = evaluate(trainSensors, trainLabels, trainSize, n);

            // Val loss
            float valLoss = 0;
            int valCount = 0;
            for (int i = trainSize; i < Math.min(trainSize + 200, n); i++) {
                float[] result = forward(trainSensors.get(i));
                valLoss += computeLoss(result, trainLabels.get(i), trainAnomalyTargets.get(i));
                valCount++;
            }
            valLoss /= Math.max(1, valCount);

            if (valAcc > bestValAcc) bestValAcc = valAcc;

            long elapsed = System.currentTimeMillis() - epochStart;

            // Progress bar
            int pct = epoch * 20 / epochs;
            StringBuilder bar = new StringBuilder("[");
            for (int i = 0; i < 20; i++) {
                if (i < pct) bar.append("=");
                else if (i == pct) bar.append(">");
                else bar.append(" ");
            }
            bar.append("]");

            System.out.printf("Epoch %2d/%d %s loss: %.4f | val_loss: %.4f | acc: %.1f%% | val_acc: %.1f%% | lr: %.5f | %dms%n",
                epoch, epochs, bar, avgLoss, valLoss, trainAcc * 100, valAcc * 100, lr, elapsed);
        }

        System.out.println();
        System.out.println("=== Training Complete ===");
        System.out.printf("Best validation accuracy: %.1f%%%n%n", bestValAcc * 100);

        // ===================== PER-CONTEXT EVALUATION =====================
        System.out.println("=== Per-Context Accuracy (Test Set) ===");

        // Generate fresh test data
        Random testRng = new Random(99);
        int[] perCtxCorrect = new int[NUM_CONTEXTS];
        int[] perCtxTotal = new int[NUM_CONTEXTS];

        for (int ctx = 0; ctx < 15; ctx++) {
            for (int i = 0; i < 50; i++) {
                float[][] sample = generateSensorSample(ctx, testRng.nextFloat());
                float[] result = forward(sample);
                int predicted = argmax(result, 0, NUM_CONTEXTS);
                perCtxTotal[ctx]++;
                if (predicted == ctx) perCtxCorrect[ctx]++;
            }
        }

        for (int ctx = 0; ctx < 15; ctx++) {
            float acc = perCtxTotal[ctx] > 0 ? perCtxCorrect[ctx] * 100f / perCtxTotal[ctx] : 0;
            int bars = (int)(acc / 5);
            StringBuilder barStr = new StringBuilder();
            for (int i = 0; i < 20; i++) barStr.append(i < bars ? "█" : "░");
            System.out.printf("  %-20s %s %.1f%%  (%d/%d)%n",
                CONTEXT_NAMES[ctx], barStr, acc, perCtxCorrect[ctx], perCtxTotal[ctx]);
        }

        // ===================== ROUTING ANALYSIS =====================
        System.out.println();
        System.out.println("=== Expert Routing Analysis ===");
        System.out.println("Which sensor experts the model activates for each context:");

        for (int ctx = 0; ctx < 15; ctx++) {
            float[] avgRouting = new float[6];
            int count = 0;
            for (int i = 0; i < 20; i++) {
                float[][] sample = generateSensorSample(ctx, testRng.nextFloat());
                float[] result = forward(sample);
                // Routing weights are stored at offset NUM_CONTEXTS + 1
                for (int e = 0; e < 6; e++) {
                    avgRouting[e] += result[NUM_CONTEXTS + 1 + e];
                }
                count++;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("  %-20s", CONTEXT_NAMES[ctx]));
            for (int e = 0; e < 6; e++) {
                int pctRouting = (int)(avgRouting[e] / count * 100);
                sb.append(String.format(" %s:%d%%", EXPERT_NAMES[e].substring(0, 3), pctRouting));
                if (e < 5) sb.append("  ");
            }
            System.out.println(sb);
        }

        // ===================== EXPORT WEIGHTS =====================
        System.out.println();
        exportWeights("trained_weights/synaptic_v1.synaptic");

        System.out.println();
        System.out.println("=== Model Summary ===");
        printModelSummary();

        System.out.println();
        System.out.println("Training complete! Weights ready for Samsung S25 Ultra deployment.");
        System.out.println();
    }

    // ===================== MODEL INIT =====================

    static void initializeModel() {
        for (int i = 0; i < 6; i++) {
            sensorProjections[i] = xavierInit(SENSOR_DIMS[i], EXPERT_DIM);
            sensorBiases[i] = new float[EXPERT_DIM];
        }
        routingGate = xavierInit(TOTAL_SENSOR_DIM, NUM_EXPERTS);
        routingBias = new float[NUM_EXPERTS];
        crossAttnQ = xavierInit(FUSED_DIM, ATTN_DIM);
        crossAttnK = xavierInit(FUSED_DIM, ATTN_DIM);
        crossAttnV = xavierInit(FUSED_DIM, ATTN_DIM);
        crossAttnOut = xavierInit(ATTN_DIM, FUSED_DIM);
        ffnW1 = xavierInit(FUSED_DIM, FFN_DIM);
        ffnB1 = new float[FFN_DIM];
        ffnW2 = xavierInit(FFN_DIM, FUSED_DIM);
        ffnB2 = new float[FUSED_DIM];
        contextHead = xavierInit(FUSED_DIM, NUM_CONTEXTS);
        contextBias = new float[NUM_CONTEXTS];
        anomalyHead = xavierInit(FUSED_DIM, 1);
        anomalyBias = new float[1];

        allParams = new ArrayList<>(Arrays.asList(
            sensorProjections[0], sensorBiases[0],
            sensorProjections[1], sensorBiases[1],
            sensorProjections[2], sensorBiases[2],
            sensorProjections[3], sensorBiases[3],
            sensorProjections[4], sensorBiases[4],
            sensorProjections[5], sensorBiases[5],
            routingGate, routingBias,
            crossAttnQ, crossAttnK, crossAttnV, crossAttnOut,
            ffnW1, ffnB1, ffnW2, ffnB2,
            contextHead, contextBias,
            anomalyHead, anomalyBias
        ));

        mState = new ArrayList<>();
        vState = new ArrayList<>();
        for (float[] p : allParams) {
            mState.add(new float[p.length]);
            vState.add(new float[p.length]);
        }
    }

    static float[] xavierInit(int fanIn, int fanOut) {
        float scale = (float)Math.sqrt(2.0 / (fanIn + fanOut));
        float[] arr = new float[fanIn * fanOut];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (rng.nextFloat() - 0.5f) * 2 * scale;
        }
        return arr;
    }

    // ===================== FORWARD PASS =====================

    /**
     * Forward pass. Returns array of:
     * [0..15] = context logits
     * [16] = anomaly score
     * [17..22] = routing weights
     * [23..214] = fused embedding (192 dims)
     */
    static float[] forward(float[][] sensorData) {
        // Concatenate all sensors for routing, normalized per-modality to prevent
        // larger-dimensional sensors (visual=64) from dominating the routing gate
        float[] allSensors = new float[TOTAL_SENSOR_DIM];
        int off = 0;
        for (int s = 0; s < 6; s++) {
            float norm = 0;
            for (float v : sensorData[s]) norm += v * v;
            norm = (float)Math.sqrt(norm) + 1e-6f;
            for (int i = 0; i < sensorData[s].length; i++) {
                allSensors[off + i] = sensorData[s][i] / norm;
            }
            off += sensorData[s].length;
        }

        // Compute routing logits
        float[] routeLogits = new float[NUM_EXPERTS];
        for (int j = 0; j < NUM_EXPERTS; j++) {
            float sum = routingBias[j];
            for (int i = 0; i < TOTAL_SENSOR_DIM; i++) {
                sum += allSensors[i] * routingGate[i * NUM_EXPERTS + j];
            }
            routeLogits[j] = sum;
        }
        float[] routeWeights = softmax(routeLogits);

        // Project each sensor through its expert
        float[] fused = new float[FUSED_DIM];
        for (int e = 0; e < 6; e++) {
            float[] input = sensorData[e];
            int sDim = SENSOR_DIMS[e];
            float[] proj = sensorProjections[e];
            float[] bias = sensorBiases[e];

            for (int j = 0; j < EXPERT_DIM; j++) {
                float sum = bias[j];
                for (int i = 0; i < sDim; i++) {
                    sum += input[i] * proj[i * EXPERT_DIM + j];
                }
                fused[e * EXPERT_DIM + j] = silu(sum) * routeWeights[e];
            }
        }

        // Cross-attention
        float[] q = matVec(crossAttnQ, fused, FUSED_DIM, ATTN_DIM);
        float[] k = matVec(crossAttnK, fused, FUSED_DIM, ATTN_DIM);
        float[] v = matVec(crossAttnV, fused, FUSED_DIM, ATTN_DIM);

        float attnScore = 0;
        float scale = 1f / (float)Math.sqrt(ATTN_DIM);
        for (int i = 0; i < ATTN_DIM; i++) attnScore += q[i] * k[i];
        attnScore *= scale;
        float attnWeight = sigmoid(attnScore);

        float[] attnOut = new float[ATTN_DIM];
        for (int i = 0; i < ATTN_DIM; i++) attnOut[i] = v[i] * attnWeight;
        float[] projected = matVec(crossAttnOut, attnOut, ATTN_DIM, FUSED_DIM);

        // Residual
        float[] postAttn = new float[FUSED_DIM];
        for (int i = 0; i < FUSED_DIM; i++) postAttn[i] = fused[i] + projected[i];

        // FFN
        float[] hidden = new float[FFN_DIM];
        for (int j = 0; j < FFN_DIM; j++) {
            float sum = ffnB1[j];
            for (int i = 0; i < FUSED_DIM; i++) sum += postAttn[i] * ffnW1[i * FFN_DIM + j];
            hidden[j] = silu(sum);
        }
        float[] ffnOut = new float[FUSED_DIM];
        for (int j = 0; j < FUSED_DIM; j++) {
            float sum = ffnB2[j];
            for (int i = 0; i < FFN_DIM; i++) sum += hidden[i] * ffnW2[i * FUSED_DIM + j];
            ffnOut[j] = sum;
        }

        // Residual
        float[] output = new float[FUSED_DIM];
        for (int i = 0; i < FUSED_DIM; i++) output[i] = postAttn[i] + ffnOut[i];

        // Context logits
        float[] ctxLogits = new float[NUM_CONTEXTS];
        for (int j = 0; j < NUM_CONTEXTS; j++) {
            float sum = contextBias[j];
            for (int i = 0; i < FUSED_DIM; i++) sum += output[i] * contextHead[i * NUM_CONTEXTS + j];
            ctxLogits[j] = sum;
        }

        // Anomaly score
        float anomLogit = anomalyBias[0];
        for (int i = 0; i < FUSED_DIM; i++) anomLogit += output[i] * anomalyHead[i];
        float anomScore = sigmoid(anomLogit);

        // Pack results
        float[] result = new float[NUM_CONTEXTS + 1 + NUM_EXPERTS + FUSED_DIM];
        System.arraycopy(ctxLogits, 0, result, 0, NUM_CONTEXTS);
        result[NUM_CONTEXTS] = anomScore;
        System.arraycopy(routeWeights, 0, result, NUM_CONTEXTS + 1, NUM_EXPERTS);
        System.arraycopy(output, 0, result, NUM_CONTEXTS + 1 + NUM_EXPERTS, FUSED_DIM);
        return result;
    }

    // ===================== LOSS =====================

    static float computeLoss(float[] result, int targetCtx, float targetAnomaly) {
        // Cross-entropy for context
        float[] logProbs = logSoftmax(result, 0, NUM_CONTEXTS);
        float ctxLoss = -logProbs[targetCtx];

        // BCE for anomaly
        float p = Math.max(1e-7f, Math.min(1 - 1e-7f, result[NUM_CONTEXTS]));
        float anomLoss = -(targetAnomaly * (float)Math.log(p) + (1 - targetAnomaly) * (float)Math.log(1 - p));

        // Load balance
        float entropy = 0;
        for (int i = 0; i < NUM_EXPERTS; i++) {
            float w = result[NUM_CONTEXTS + 1 + i];
            if (w > 1e-7f) entropy -= w * (float)Math.log(w);
        }
        float maxEntropy = (float)Math.log(NUM_EXPERTS);
        float balanceLoss = (maxEntropy - entropy) / maxEntropy;

        return ctxLoss + 0.5f * anomLoss + 0.3f * balanceLoss;
    }

    // ===================== GRADIENT (SPSA) =====================

    static List<float[]> computeSPSAGradients(float[][] sample, int targetCtx, float targetAnomaly) {
        float eps = 0.001f;

        // Random perturbation directions
        List<float[]> perturbations = new ArrayList<>();
        for (float[] p : allParams) {
            float[] pert = new float[p.length];
            for (int i = 0; i < p.length; i++) pert[i] = rng.nextBoolean() ? 1f : -1f;
            perturbations.add(pert);
        }

        // Positive perturbation
        for (int p = 0; p < allParams.size(); p++)
            for (int i = 0; i < allParams.get(p).length; i++)
                allParams.get(p)[i] += eps * perturbations.get(p)[i];

        float[] resPlus = forward(sample);
        float lossPlus = computeLoss(resPlus, targetCtx, targetAnomaly);

        // Negative perturbation
        for (int p = 0; p < allParams.size(); p++)
            for (int i = 0; i < allParams.get(p).length; i++)
                allParams.get(p)[i] -= 2 * eps * perturbations.get(p)[i];

        float[] resMinus = forward(sample);
        float lossMinus = computeLoss(resMinus, targetCtx, targetAnomaly);

        // Restore
        for (int p = 0; p < allParams.size(); p++)
            for (int i = 0; i < allParams.get(p).length; i++)
                allParams.get(p)[i] += eps * perturbations.get(p)[i];

        // SPSA gradient
        float lossGrad = (lossPlus - lossMinus) / (2 * eps);
        List<float[]> gradients = new ArrayList<>();
        for (int p = 0; p < allParams.size(); p++) {
            float[] grad = new float[allParams.get(p).length];
            for (int i = 0; i < grad.length; i++) {
                grad[i] = lossGrad / perturbations.get(p)[i];
            }
            gradients.add(grad);
        }
        return gradients;
    }

    // ===================== ADAMW =====================

    static void adamWStep(List<float[]> gradients, float lr) {
        adamStep++;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
        float bc1 = 1 - (float)Math.pow(beta1, adamStep);
        float bc2 = 1 - (float)Math.pow(beta2, adamStep);

        for (int p = 0; p < allParams.size(); p++) {
            float[] param = allParams.get(p);
            float[] grad = gradients.get(p);
            float[] m = mState.get(p);
            float[] v = vState.get(p);

            for (int i = 0; i < param.length; i++) {
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;
                param[i] -= lr * (mHat / ((float)Math.sqrt(vHat) + eps) + wd * param[i]);
            }
        }
    }

    static void clipGradients(List<float[]> grads, float maxNorm) {
        float totalNorm = 0;
        for (float[] g : grads) for (float v : g) totalNorm += v * v;
        totalNorm = (float)Math.sqrt(totalNorm);
        if (totalNorm > maxNorm) {
            float scale = maxNorm / totalNorm;
            for (float[] g : grads) for (int i = 0; i < g.length; i++) g[i] *= scale;
        }
    }

    // ===================== EVALUATION =====================

    static float evaluate(List<float[][]> data, List<Integer> labels, int start, int end) {
        int correct = 0, total = 0;
        for (int i = start; i < end; i++) {
            float[] result = forward(data.get(i));
            int predicted = argmax(result, 0, NUM_CONTEXTS);
            if (predicted == labels.get(i)) correct++;
            total++;
        }
        return total > 0 ? (float)correct / total : 0;
    }

    // ===================== SYNTHETIC DATA =====================

    static float[][] generateSensorSample(int context, float phase) {
        float t = phase * 2 * (float)Math.PI;
        float[][] sample = new float[6][];

        // Default noise function
        switch (context) {
            case 0: // WALKING
                sample[0] = new float[]{n(.3f)+sin(t*4)*2, n(.2f)+cos(t*4)*1.5f, 9.81f+sin(t*8)*1+n(.2f),
                    sin(t*4)*.3f+n(.05f), n(.1f), sin(t*4)*.2f+n(.05f), n(.1f), n(.1f), 9.81f};
                sample[1] = new float[]{25+n(2), -10+n(2), -45+n(2), 37.77f+phase*.001f, -122.42f+phase*.001f, 10, 1013.25f+n(.1f)};
                sample[2] = genVisual(.7f, .4f, .5f);
                sample[3] = genAudio("footsteps", .3f);
                sample[4] = new float[]{.6f+n(.1f), 1f, 22+n(1), 50+n(5)};
                sample[5] = new float[]{-70+n(5), -80+n(5), -85+n(5), -90+n(5), -95+n(5), -88+n(5), n(.5f), n(.5f)};
                break;
            case 1: // SITTING_DESK
                sample[0] = new float[]{n(.05f), n(.05f), 9.81f+n(.02f), n(.01f), n(.01f), n(.01f), 0, 0, 9.81f};
                sample[1] = new float[]{30+n(.5f), -15+n(.5f), -50+n(.5f), 37.77f, -122.42f, 15, 1013.5f+n(.05f)};
                sample[2] = genVisual(.4f, .05f, .6f);
                sample[3] = genAudio("office", .15f);
                sample[4] = new float[]{.35f+n(.02f), .3f+n(.05f), 23+n(.5f), 40+n(3)};
                sample[5] = new float[]{-40+n(2), -45+n(2), -50+n(2), -60+n(3), -65+n(3), -70+n(3), .5f+n(.1f), .3f+n(.1f)};
                break;
            case 2: // DRIVING
                sample[0] = new float[]{n(.3f)+sin(t*.5f)*.5f, n(.2f)+cos(t*.3f)*.3f, 9.81f+n(.15f),
                    n(.05f), sin(t*.5f)*.1f, n(.02f), sin(t*.5f)*.3f, 0, 9.81f};
                sample[1] = new float[]{20+n(5), -20+n(5), -40+n(5), 37.77f+phase*.01f, -122.42f+phase*.008f, 10, 1013.25f+n(.3f)};
                sample[2] = genVisual(.5f, .6f, .4f);
                sample[3] = genAudio("driving", .4f);
                sample[4] = new float[]{.5f+n(.15f), .2f+n(.1f), 24+n(2), 45+n(5)};
                sample[5] = new float[]{-75+n(10), -80+n(10), -85+n(10), -50+n(3), -55+n(3), -60+n(3), n(.3f), n(.3f)};
                break;
            case 3: // IN_MEETING
                sample[0] = new float[]{n(.08f), n(.08f), 9.81f+n(.03f), n(.02f), n(.02f), n(.02f), 0, 0, 9.81f};
                sample[1] = new float[]{35+n(1), -12+n(1), -48+n(1), 37.77f, -122.42f, 15, 1013.4f+n(.05f)};
                sample[2] = genVisual(.35f, .08f, .3f);
                sample[3] = genAudio("speech", .5f);
                sample[4] = new float[]{.3f+n(.02f), .8f+n(.1f), 23+n(.3f), 42+n(2)};
                sample[5] = new float[]{-35+n(2), -40+n(2), -42+n(2), -55+n(3), -58+n(3), -60+n(3), .8f+n(.1f), .6f+n(.1f)};
                break;
            case 4: // SLEEPING
                sample[0] = new float[]{n(.01f), n(.01f), 9.81f+n(.005f), n(.002f), n(.002f), n(.002f), n(.5f), 9.5f+n(.3f), n(.5f)};
                sample[1] = new float[]{28+n(.2f), -18+n(.2f), -52+n(.2f), 37.77f, -122.42f, 10, 1013.3f+n(.02f)};
                sample[2] = genVisual(.02f, 0, .01f);
                sample[3] = genAudio("silence", .02f);
                sample[4] = new float[]{.01f+n(.005f), 1f, 20+n(.2f), 55+n(2)};
                sample[5] = new float[]{-45+n(1), -50+n(1), -55+n(1), -70+n(2), -75+n(2), -80+n(2), n(.1f), n(.1f)};
                break;
            case 5: // COOKING
                sample[0] = new float[]{n(.5f)+sin(t*3)*.3f, n(.4f)+cos(t*2)*.4f, 9.81f+n(.3f), n(.2f), n(.15f), n(.3f), n(.2f), n(.2f), 9.81f};
                sample[1] = new float[]{32+n(1), -14+n(1), -46+n(1), 37.77f, -122.42f, 10, 1013.3f+n(.1f)};
                sample[2] = genVisual(.5f, .25f, .45f);
                sample[3] = genAudio("kitchen", .35f);
                sample[4] = new float[]{.45f+n(.05f), 1f, 26+n(2), 60+n(5)};
                sample[5] = new float[]{-42+n(2), -48+n(2), -52+n(2), -65+n(3), -70+n(3), -75+n(3), n(.2f), n(.2f)};
                break;
            case 6: // EXERCISING
                sample[0] = new float[]{sin(t*6)*4+n(1), cos(t*6)*3+n(.8f), 9.81f+sin(t*12)*3+n(.5f),
                    sin(t*6)*1+n(.3f), n(.5f), cos(t*6)*.8f+n(.2f), n(.5f), n(.5f), 9.81f};
                sample[1] = new float[]{25+n(3), -10+n(3), -45+n(3), 37.77f+phase*.002f, -122.42f+phase*.002f, 10, 1013.2f+n(.15f)};
                sample[2] = genVisual(.6f, .7f, .4f);
                sample[3] = genAudio("exercise", .45f);
                sample[4] = new float[]{.55f+n(.1f), .5f+n(.3f), 28+n(2), 65+n(5)};
                sample[5] = new float[]{-65+n(8), -70+n(8), -75+n(8), -45+n(3), -50+n(3), -55+n(3), n(.3f), n(.3f)};
                break;
            case 7: // ELEVATOR
                sample[0] = new float[]{n(.05f), n(.05f), 9.81f+sin(t*.5f)*.8f+n(.05f), n(.02f), n(.02f), n(.01f), 0, 0, 9.81f+sin(t*.5f)*.5f};
                sample[1] = new float[]{40+n(3), -8+n(3), -55+n(3), 37.77f, -122.42f, 15+phase*20, 1013.25f-phase*3};
                sample[2] = genVisual(.25f, .02f, .2f);
                sample[3] = genAudio("silence", .1f);
                sample[4] = new float[]{.2f+n(.02f), .4f+n(.1f), 22+n(.5f), 38+n(2)};
                sample[5] = new float[]{-60+n(5), -65+n(5), -70+n(5), -70+n(5), -75+n(5), -80+n(5), n(.2f), n(.2f)};
                break;
            case 8: // PHONE_CALL
                sample[0] = new float[]{n(.1f), n(.1f), 9.81f+n(.05f), n(.05f), n(.05f), n(.03f), 3+n(.5f), 8+n(.5f), 4+n(.5f)};
                sample[1] = new float[]{30+n(1), -15+n(1), -48+n(1), 37.77f, -122.42f, 10, 1013.3f+n(.05f)};
                sample[2] = genVisual(.1f, .01f, .05f);
                sample[3] = genAudio("speech", .6f);
                sample[4] = new float[]{.05f+n(.02f), 0f, 30+n(1), 50+n(3)};
                sample[5] = new float[]{-45+n(3), -50+n(3), -55+n(3), -60+n(3), -65+n(3), -70+n(3), n(.2f), n(.2f)};
                break;
            case 9: // OUTDOORS_NATURE
                sample[0] = new float[]{sin(t*3)*1+n(.3f), cos(t*3)*.8f+n(.3f), 9.81f+sin(t*6)*.5f+n(.3f),
                    n(.1f), n(.1f), n(.1f), n(.2f), n(.2f), 9.81f};
                sample[1] = new float[]{22+n(1), -12+n(1), -42+n(1), 37.78f+phase*.003f, -122.42f+phase*.003f, 50+n(5), 1012+n(.5f)};
                sample[2] = genVisual(.8f, .3f, .35f);
                sample[3] = genAudio("nature", .2f);
                sample[4] = new float[]{.85f+n(.1f), 1f, 25+n(3), 55+n(10)};
                sample[5] = new float[]{-90+n(3), -92+n(3), -95+n(3), -85+n(5), -88+n(5), -90+n(5), n(.1f), n(.1f)};
                break;
            case 10: // PUBLIC_TRANSIT
                sample[0] = new float[]{n(.4f)+sin(t)*.3f, n(.3f)+cos(t*.5f)*.5f, 9.81f+n(.2f),
                    n(.08f), sin(t*.5f)*.1f, n(.05f), sin(t*.3f)*.2f, 0, 9.81f};
                sample[1] = new float[]{35+n(4), -15+n(4), -50+n(4), 37.77f+phase*.005f, -122.42f+phase*.004f, 10, 1013.2f+n(.1f)};
                sample[2] = genVisual(.3f, .15f, .35f);
                sample[3] = genAudio("crowd", .4f);
                sample[4] = new float[]{.25f+n(.05f), .5f+n(.2f), 24+n(1), 45+n(3)};
                sample[5] = new float[]{-55+n(5), -60+n(5), -65+n(5), -50+n(5), -55+n(5), -60+n(5), n(.3f), n(.3f)};
                break;
            case 11: // SHOPPING
                sample[0] = new float[]{sin(t*3)*1.5f+n(.3f), cos(t*3)+n(.2f), 9.81f+sin(t*6)*.7f+n(.2f),
                    n(.1f), n(.1f), n(.15f), n(.1f), n(.1f), 9.81f};
                sample[1] = new float[]{38+n(2), -10+n(2), -52+n(2), 37.77f+phase*.0005f, -122.42f+phase*.0005f, 12, 1013.5f+n(.05f)};
                sample[2] = genVisual(.5f, .3f, .6f);
                sample[3] = genAudio("crowd", .35f);
                sample[4] = new float[]{.55f+n(.05f), 1f, 22+n(1), 40+n(3)};
                sample[5] = new float[]{-40+n(3), -45+n(3), -48+n(3), -45+n(3), -48+n(3), -50+n(3), .6f+n(.2f), .4f+n(.2f)};
                break;
            case 12: // CLIMBING_STAIRS
                sample[0] = new float[]{sin(t*5)*1.5f+n(.4f), cos(t*5)+n(.3f), 9.81f+sin(t*10)*2+n(.3f),
                    sin(t*5)*.3f+n(.1f), n(.1f), n(.1f), n(.15f), n(.15f), 9.81f};
                sample[1] = new float[]{33+n(2), -14+n(2), -49+n(2), 37.77f, -122.42f, 15+phase*10, 1013.25f-phase*1.5f};
                sample[2] = genVisual(.3f, .35f, .5f);
                sample[3] = genAudio("footsteps", .25f);
                sample[4] = new float[]{.25f+n(.05f), 1f, 23+n(1), 42+n(3)};
                sample[5] = new float[]{-50+n(5), -55+n(5), -60+n(5), -65+n(3), -70+n(3), -75+n(3), n(.2f), n(.2f)};
                break;
            case 13: // CYCLING
                sample[0] = new float[]{sin(t*4)*.8f+n(.5f), cos(t*4)*.5f+n(.3f), 9.81f+n(.4f),
                    sin(t*4)*.2f+n(.1f), n(.15f), n(.1f), sin(t*.3f)*.5f, 0, 9.81f};
                sample[1] = new float[]{24+n(2), -11+n(2), -43+n(2), 37.77f+phase*.008f, -122.42f+phase*.006f, 10+n(2), 1013.1f+n(.3f)};
                sample[2] = genVisual(.7f, .55f, .4f);
                sample[3] = genAudio("wind", .35f);
                sample[4] = new float[]{.75f+n(.1f), 1f, 24+n(3), 50+n(8)};
                sample[5] = new float[]{-80+n(8), -85+n(8), -88+n(8), -50+n(3), -55+n(3), -60+n(3), n(.2f), n(.2f)};
                break;
            case 14: // IDLE_TABLE
                sample[0] = new float[]{n(.01f), n(.01f), 9.81f+n(.005f), n(.002f), n(.002f), n(.002f), 0, 0, 9.81f};
                sample[1] = new float[]{30+n(.3f), -14+n(.3f), -48+n(.3f), 37.77f, -122.42f, 10, 1013.4f+n(.02f)};
                sample[2] = genVisual(.35f, .01f, .2f);
                sample[3] = genAudio("silence", .05f);
                sample[4] = new float[]{.3f+n(.02f), .25f+n(.05f), 22+n(.5f), 42+n(2)};
                sample[5] = new float[]{-42+n(1), -47+n(1), -52+n(1), -62+n(2), -67+n(2), -72+n(2), n(.1f), n(.1f)};
                break;
            default: // TRANSITION - blend of two random contexts
                float[][] a = generateSensorSample(rng.nextInt(15), phase);
                float[][] b = generateSensorSample(rng.nextInt(15), phase);
                sample = new float[6][];
                for (int s = 0; s < 6; s++) {
                    sample[s] = new float[a[s].length];
                    float blend = phase;
                    for (int i = 0; i < a[s].length; i++)
                        sample[s][i] = a[s][i] * (1 - blend) + b[s][i] * blend;
                }
                break;
        }
        return sample;
    }

    static float[][] generateAnomalousSample(int context) {
        float[][] sample = generateSensorSample(context, rng.nextFloat());
        int anomType = rng.nextInt(4);
        switch (anomType) {
            case 0: for (int i = 0; i < sample[0].length; i++) sample[0][i] *= 2 + rng.nextFloat() * 3; break;
            case 1: for (int i = 0; i < sample[3].length; i++) sample[3][i] *= 3 + rng.nextFloat() * 2; break;
            case 2: for (int i = 0; i < sample[4].length; i++) sample[4][i] *= rng.nextFloat() * 0.1f; break;
            case 3: sample[1][6] += (rng.nextFloat() - 0.5f) * 20; break;
        }
        return sample;
    }

    static float[] genVisual(float brightness, float motion, float edges) {
        float[] v = new float[64];
        for (int i = 0; i < 64; i++) {
            if (i < 8) v[i] = brightness + n(.1f);
            else if (i < 20) v[i] = Math.max(0, Math.min(1, (1f / (1 + Math.abs(i - 8 - brightness * 12))) + n(.05f)));
            else if (i < 28) v[i] = motion + n(.05f);
            else if (i < 44) v[i] = edges + n(.1f);
            else if (i < 52) v[i] = brightness > .1f ? .5f + n(.15f) : n(.02f);
            else v[i] = edges * .8f + n(.08f);
            v[i] = Math.max(0, Math.min(1, v[i]));
        }
        return v;
    }

    static float[] genAudio(String type, float energy) {
        float[] a = new float[40];
        for (int i = 0; i < 40; i++) {
            float base;
            switch (type) {
                case "footsteps": base = i < 5 ? energy * 1.5f : energy * .3f * (float)Math.exp(-i * .1); break;
                case "office": base = energy * .5f * (float)Math.exp(-i * .05) + (i >= 3 && i <= 8 ? .1f : 0); break;
                case "driving": base = energy * (i < 3 ? 1.2f : .6f) * (float)Math.exp(-i * .03); break;
                case "speech": base = i >= 2 && i <= 15 ? energy * .8f : energy * .1f; break;
                case "silence": base = energy * .1f * (float)Math.exp(-i * .2); break;
                case "kitchen": base = energy * .7f * (float)Math.exp(-i * .04) + (i >= 5 && i <= 12 ? .15f : 0); break;
                case "exercise": base = energy * (i < 8 ? 1f : .5f) * (float)Math.exp(-i * .05); break;
                case "nature": base = energy * .6f * (float)Math.exp(-i * .06); break;
                case "crowd": base = energy * .8f * (float)Math.exp(-i * .04); break;
                case "wind": base = i < 4 ? energy * 1.5f : energy * .2f * (float)Math.exp(-i * .1); break;
                default: base = energy * .5f * (float)Math.exp(-i * .05); break;
            }
            a[i] = Math.max(-5, Math.min(5, base + n(energy * .1f)));
        }
        return a;
    }

    // ===================== EXPORT =====================

    static void exportWeights(String path) {
        try {
            new File("trained_weights").mkdirs();
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)));

            // Magic + version
            dos.writeBytes("SYNP");
            dos.writeInt(1);

            String[] names = {
                "expert.motion.proj", "expert.motion.bias",
                "expert.spatial.proj", "expert.spatial.bias",
                "expert.visual.proj", "expert.visual.bias",
                "expert.audio.proj", "expert.audio.bias",
                "expert.ambient.proj", "expert.ambient.bias",
                "expert.radio.proj", "expert.radio.bias",
                "router.gate", "router.bias",
                "attn.q", "attn.k", "attn.v", "attn.out",
                "ffn.w1", "ffn.b1", "ffn.w2", "ffn.b2",
                "head.context", "head.context_bias",
                "head.anomaly", "head.anomaly_bias"
            };

            dos.writeInt(allParams.size());

            long totalBytes = 12;
            for (int p = 0; p < allParams.size(); p++) {
                byte[] nameBytes = names[p].getBytes("UTF-8");
                dos.writeShort(nameBytes.length);
                dos.write(nameBytes);
                dos.writeByte(1); // 1 dimension (flat)
                dos.writeInt(allParams.get(p).length);

                ByteBuffer buf = ByteBuffer.allocate(allParams.get(p).length * 4).order(ByteOrder.LITTLE_ENDIAN);
                for (float v : allParams.get(p)) buf.putFloat(v);
                dos.write(buf.array());

                totalBytes += 2 + nameBytes.length + 1 + 4 + allParams.get(p).length * 4;
            }
            dos.close();
            System.out.println("Exported weights to " + path + " (" + (totalBytes / 1024) + "KB)");
        } catch (Exception e) {
            System.err.println("Export failed: " + e.getMessage());
        }
    }

    static void printModelSummary() {
        String[] names = {
            "expert.motion.proj", "expert.motion.bias", "expert.spatial.proj", "expert.spatial.bias",
            "expert.visual.proj", "expert.visual.bias", "expert.audio.proj", "expert.audio.bias",
            "expert.ambient.proj", "expert.ambient.bias", "expert.radio.proj", "expert.radio.bias",
            "router.gate", "router.bias", "attn.q", "attn.k", "attn.v", "attn.out",
            "ffn.w1", "ffn.b1", "ffn.w2", "ffn.b2",
            "head.context", "head.context_bias", "head.anomaly", "head.anomaly_bias"
        };
        int total = 0;
        System.out.printf("%-30s %s%n", "Layer", "Params");
        System.out.println("-".repeat(50));
        for (int i = 0; i < allParams.size(); i++) {
            int count = allParams.get(i).length;
            total += count;
            System.out.printf("%-30s %,d%n", names[i], count);
        }
        System.out.println("-".repeat(50));
        System.out.printf("%-30s %,d%n", "TOTAL", total);
        System.out.printf("Weight file size: %dKB%n", total * 4 / 1024);
    }

    // ===================== MATH HELPERS =====================

    static float n(float scale) { return (rng.nextFloat() - 0.5f) * 2 * scale; }
    static float sin(float x) { return (float)Math.sin(x); }
    static float cos(float x) { return (float)Math.cos(x); }
    static float silu(float x) { return x * sigmoid(x); }
    static float sigmoid(float x) { return 1f / (1f + (float)Math.exp(-x)); }

    static float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) max = Math.max(max, v);
        float sum = 0;
        float[] r = new float[x.length];
        for (int i = 0; i < x.length; i++) { r[i] = (float)Math.exp(x[i] - max); sum += r[i]; }
        for (int i = 0; i < r.length; i++) r[i] /= sum;
        return r;
    }

    static float[] logSoftmax(float[] result, int offset, int len) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) max = Math.max(max, result[offset + i]);
        float sumExp = 0;
        for (int i = 0; i < len; i++) sumExp += (float)Math.exp(result[offset + i] - max);
        float logSumExp = max + (float)Math.log(sumExp);
        float[] r = new float[len];
        for (int i = 0; i < len; i++) r[i] = result[offset + i] - logSumExp;
        return r;
    }

    static float[] matVec(float[] mat, float[] vec, int rows, int cols) {
        float[] r = new float[cols];
        for (int j = 0; j < cols; j++) {
            float sum = 0;
            for (int i = 0; i < rows; i++) sum += vec[i] * mat[i * cols + j];
            r[j] = sum;
        }
        return r;
    }

    static int argmax(float[] arr, int offset, int len) {
        int best = 0;
        float bestVal = arr[offset];
        for (int i = 1; i < len; i++) {
            if (arr[offset + i] > bestVal) { bestVal = arr[offset + i]; best = i; }
        }
        return best;
    }
}
