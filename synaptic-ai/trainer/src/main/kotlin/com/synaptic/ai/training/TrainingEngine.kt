package com.synaptic.ai.training

import com.synaptic.ai.data.SyntheticDataGenerator
import com.synaptic.ai.data.SyntheticDataGenerator.ActivityContext
import kotlin.math.*
import kotlin.random.Random

/**
 * TrainingEngine: Trains the SynapticAI model on synthetic sensor data.
 *
 * Since we can't use PyTorch/TensorFlow in a pure Kotlin/JVM environment,
 * this implements a simplified but functional training pipeline:
 *
 * 1. Forward pass through a simplified version of the model
 * 2. Multi-task loss computation (context classification + anomaly detection + routing balance)
 * 3. Numerical gradient estimation (for this small model, it's tractable)
 * 4. AdamW parameter updates
 *
 * The model we train here is a streamlined version focused on learning:
 * - Sensor-to-embedding projections (what each sensor modality means)
 * - Cross-sensor attention patterns (which sensors correlate)
 * - Context classification (what activity is happening)
 * - Anomaly detection (is this unusual)
 * - Expert routing preferences (which experts matter for which context)
 */
class TrainingEngine {

    // Simplified trainable model: sensor projections + classification head
    // This captures the most important learnable components
    class TrainableModel(
        // 6 sensor projection matrices: project raw sensor -> 32-dim embedding each
        val sensorProjections: Array<FloatArray>,  // [6][sensor_dim * 32]
        val sensorBiases: Array<FloatArray>,       // [6][32]

        // Expert routing gate: [total_sensor_dim, 6]
        val routingGate: FloatArray,               // [132 * 6]
        val routingBias: FloatArray,               // [6]

        // Cross-sensor attention: [192, 192] (6 experts * 32 dim = 192)
        val crossAttentionQ: FloatArray,           // [192 * 64]
        val crossAttentionK: FloatArray,           // [192 * 64]
        val crossAttentionV: FloatArray,           // [192 * 64]
        val crossAttentionOut: FloatArray,         // [64 * 192]

        // Feed-forward after attention
        val ffnWeight1: FloatArray,                // [192 * 128]
        val ffnBias1: FloatArray,                  // [128]
        val ffnWeight2: FloatArray,                // [128 * 192]
        val ffnBias2: FloatArray,                  // [192]

        // Context classification head: [192, 16]
        val contextHead: FloatArray,               // [192 * 16]
        val contextBias: FloatArray,               // [16]

        // Anomaly detection head: [192, 1]
        val anomalyHead: FloatArray,               // [192]
        val anomalyBias: FloatArray,               // [1]
    ) {
        fun allParameters(): List<FloatArray> {
            return sensorProjections.toList() + sensorBiases.toList() + listOf(
                routingGate, routingBias,
                crossAttentionQ, crossAttentionK, crossAttentionV, crossAttentionOut,
                ffnWeight1, ffnBias1, ffnWeight2, ffnBias2,
                contextHead, contextBias,
                anomalyHead, anomalyBias,
            )
        }

        fun totalParams(): Int = allParameters().sumOf { it.size }

        companion object {
            val SENSOR_DIMS = intArrayOf(9, 7, 64, 40, 4, 8)  // Motion, Spatial, Visual, Audio, Ambient, Radio
            const val EXPERT_DIM = 32
            const val FUSED_DIM = 192  // 6 * 32
            const val ATTN_DIM = 64
            const val FFN_DIM = 128
            const val NUM_CONTEXTS = 16
            const val TOTAL_SENSOR_DIM = 132

            fun initialize(rng: Random = Random(42)): TrainableModel {
                fun randArray(size: Int, scale: Float = 0.02f): FloatArray {
                    return FloatArray(size) { (rng.nextFloat() - 0.5f) * 2 * scale }
                }

                // Xavier initialization for better gradient flow
                fun xavierArray(fanIn: Int, fanOut: Int): FloatArray {
                    val scale = sqrt(2.0f / (fanIn + fanOut))
                    return FloatArray(fanIn * fanOut) { (rng.nextFloat() - 0.5f) * 2 * scale }
                }

                return TrainableModel(
                    sensorProjections = Array(6) { i -> xavierArray(SENSOR_DIMS[i], EXPERT_DIM) },
                    sensorBiases = Array(6) { FloatArray(EXPERT_DIM) },
                    routingGate = xavierArray(TOTAL_SENSOR_DIM, 6),
                    routingBias = FloatArray(6),
                    crossAttentionQ = xavierArray(FUSED_DIM, ATTN_DIM),
                    crossAttentionK = xavierArray(FUSED_DIM, ATTN_DIM),
                    crossAttentionV = xavierArray(FUSED_DIM, ATTN_DIM),
                    crossAttentionOut = xavierArray(ATTN_DIM, FUSED_DIM),
                    ffnWeight1 = xavierArray(FUSED_DIM, FFN_DIM),
                    ffnBias1 = FloatArray(FFN_DIM),
                    ffnWeight2 = xavierArray(FFN_DIM, FUSED_DIM),
                    ffnBias2 = FloatArray(FUSED_DIM),
                    contextHead = xavierArray(FUSED_DIM, NUM_CONTEXTS),
                    contextBias = FloatArray(NUM_CONTEXTS),
                    anomalyHead = xavierArray(FUSED_DIM, 1),
                    anomalyBias = FloatArray(1),
                )
            }
        }
    }

    /**
     * Forward pass through the trainable model.
     * Returns logits for context classification and anomaly score.
     */
    data class ForwardResult(
        val contextLogits: FloatArray,    // [16]
        val anomalyScore: Float,          // sigmoid output
        val routingWeights: FloatArray,   // [6] - which experts were weighted how
        val fusedEmbedding: FloatArray,   // [192] - the learned representation
    )

    fun forward(model: TrainableModel, sample: SyntheticDataGenerator.SensorSample): ForwardResult {
        val sensorArrays = arrayOf(
            sample.motion, sample.spatial, sample.visual,
            sample.audio, sample.ambient, sample.radio,
        )

        // Step 1: Compute routing weights from concatenated sensors
        val allSensors = FloatArray(TrainableModel.TOTAL_SENSOR_DIM)
        var offset = 0
        for (arr in sensorArrays) {
            System.arraycopy(arr, 0, allSensors, offset, arr.size)
            offset += arr.size
        }

        val routingLogits = FloatArray(6)
        for (j in 0 until 6) {
            var sum = model.routingBias[j]
            for (i in 0 until TrainableModel.TOTAL_SENSOR_DIM) {
                sum += allSensors[i] * model.routingGate[i * 6 + j]
            }
            routingLogits[j] = sum
        }
        val routingWeights = softmax(routingLogits)

        // Step 2: Project each sensor through its expert
        val expertOutputs = Array(6) { expertIdx ->
            val sensorData = sensorArrays[expertIdx]
            val sensorDim = TrainableModel.SENSOR_DIMS[expertIdx]
            val proj = model.sensorProjections[expertIdx]
            val bias = model.sensorBiases[expertIdx]

            FloatArray(TrainableModel.EXPERT_DIM) { j ->
                var sum = bias[j]
                for (i in 0 until sensorDim) {
                    sum += sensorData[i] * proj[i * TrainableModel.EXPERT_DIM + j]
                }
                silu(sum) // SiLU activation
            }
        }

        // Step 3: Weight and concatenate expert outputs
        val fused = FloatArray(TrainableModel.FUSED_DIM)
        for (e in 0 until 6) {
            val weight = routingWeights[e]
            for (j in 0 until TrainableModel.EXPERT_DIM) {
                fused[e * TrainableModel.EXPERT_DIM + j] = expertOutputs[e][j] * weight
            }
        }

        // Step 4: Cross-sensor self-attention
        val q = matVecMul(model.crossAttentionQ, fused, TrainableModel.FUSED_DIM, TrainableModel.ATTN_DIM)
        val k = matVecMul(model.crossAttentionK, fused, TrainableModel.FUSED_DIM, TrainableModel.ATTN_DIM)
        val v = matVecMul(model.crossAttentionV, fused, TrainableModel.FUSED_DIM, TrainableModel.ATTN_DIM)

        // Scaled dot-product (simplified single-head for training efficiency)
        val scale = 1f / sqrt(TrainableModel.ATTN_DIM.toFloat())
        var attnScore = 0f
        for (i in 0 until TrainableModel.ATTN_DIM) {
            attnScore += q[i] * k[i]
        }
        attnScore *= scale
        val attnWeight = 1f / (1f + exp(-attnScore)) // sigmoid attention

        val attnOut = FloatArray(TrainableModel.ATTN_DIM) { i -> v[i] * attnWeight }
        val projected = matVecMul(model.crossAttentionOut, attnOut, TrainableModel.ATTN_DIM, TrainableModel.FUSED_DIM)

        // Residual connection
        val postAttn = FloatArray(TrainableModel.FUSED_DIM) { i -> fused[i] + projected[i] }

        // Step 5: Feed-forward network with SiLU
        val ffnHidden = FloatArray(TrainableModel.FFN_DIM) { j ->
            var sum = model.ffnBias1[j]
            for (i in 0 until TrainableModel.FUSED_DIM) {
                sum += postAttn[i] * model.ffnWeight1[i * TrainableModel.FFN_DIM + j]
            }
            silu(sum)
        }

        val ffnOut = FloatArray(TrainableModel.FUSED_DIM) { j ->
            var sum = model.ffnBias2[j]
            for (i in 0 until TrainableModel.FFN_DIM) {
                sum += ffnHidden[i] * model.ffnWeight2[i * TrainableModel.FUSED_DIM + j]
            }
            sum
        }

        // Residual connection
        val output = FloatArray(TrainableModel.FUSED_DIM) { i -> postAttn[i] + ffnOut[i] }

        // Step 6: Classification head
        val contextLogits = FloatArray(TrainableModel.NUM_CONTEXTS) { j ->
            var sum = model.contextBias[j]
            for (i in 0 until TrainableModel.FUSED_DIM) {
                sum += output[i] * model.contextHead[i * TrainableModel.NUM_CONTEXTS + j]
            }
            sum
        }

        // Step 7: Anomaly head
        var anomalyLogit = model.anomalyBias[0]
        for (i in 0 until TrainableModel.FUSED_DIM) {
            anomalyLogit += output[i] * model.anomalyHead[i]
        }
        val anomalyScore = 1f / (1f + exp(-anomalyLogit))

        return ForwardResult(
            contextLogits = contextLogits,
            anomalyScore = anomalyScore,
            routingWeights = routingWeights,
            fusedEmbedding = output,
        )
    }

    /**
     * Compute multi-task loss:
     * 1. Cross-entropy loss for context classification
     * 2. Binary cross-entropy for anomaly detection
     * 3. Load balancing loss for routing (encourage all experts to be used)
     */
    fun computeLoss(
        result: ForwardResult,
        targetContext: Int,
        targetAnomaly: Float,
    ): Float {
        // Cross-entropy loss for context classification
        val logProbs = logSoftmax(result.contextLogits)
        val contextLoss = -logProbs[targetContext]

        // Binary cross-entropy for anomaly detection
        val p = result.anomalyScore.coerceIn(1e-7f, 1f - 1e-7f)
        val anomalyLoss = -(targetAnomaly * ln(p) + (1 - targetAnomaly) * ln(1 - p))

        // Load balancing: penalize routing entropy being too low (experts too specialized)
        var routingEntropy = 0f
        for (w in result.routingWeights) {
            if (w > 1e-7f) routingEntropy -= w * ln(w)
        }
        val maxEntropy = ln(6f)
        val balanceLoss = (maxEntropy - routingEntropy) / maxEntropy  // 0 = perfectly balanced

        // Combined loss (weighted)
        return contextLoss + 0.5f * anomalyLoss + 0.1f * balanceLoss
    }

    /**
     * Compute accuracy on a batch of samples.
     */
    fun evaluateAccuracy(
        model: TrainableModel,
        samples: List<SyntheticDataGenerator.TrainingSample>,
    ): Pair<Float, Float> {
        var correct = 0
        var totalAnomalyError = 0f

        for (sample in samples) {
            val sensorSample = sample.sensorSequence[sample.sensorSequence.size / 2] // Use middle frame
            val result = forward(model, sensorSample)

            val predicted = result.contextLogits.indices.maxByOrNull { result.contextLogits[it] } ?: 0
            if (predicted == sample.contextLabel) correct++

            totalAnomalyError += abs(result.anomalyScore - sample.anomalyScore)
        }

        val accuracy = correct.toFloat() / samples.size
        val avgAnomalyError = totalAnomalyError / samples.size
        return accuracy to avgAnomalyError
    }

    // --- Utility functions ---

    private fun silu(x: Float): Float = x * (1f / (1f + exp(-x)))

    private fun softmax(x: FloatArray): FloatArray {
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in x) maxVal = maxOf(maxVal, v)
        var sumExp = 0f
        val result = FloatArray(x.size)
        for (i in x.indices) {
            result[i] = exp(x[i] - maxVal)
            sumExp += result[i]
        }
        for (i in result.indices) result[i] /= sumExp
        return result
    }

    private fun logSoftmax(x: FloatArray): FloatArray {
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in x) maxVal = maxOf(maxVal, v)
        var sumExp = 0f
        for (v in x) sumExp += exp(v - maxVal)
        val logSumExp = maxVal + ln(sumExp)
        return FloatArray(x.size) { i -> x[i] - logSumExp }
    }

    private fun matVecMul(mat: FloatArray, vec: FloatArray, rows: Int, cols: Int): FloatArray {
        return FloatArray(cols) { j ->
            var sum = 0f
            for (i in 0 until rows) {
                sum += vec[i] * mat[i * cols + j]
            }
            sum
        }
    }
}
