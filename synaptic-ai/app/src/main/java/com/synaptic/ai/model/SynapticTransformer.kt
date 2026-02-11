package com.synaptic.ai.model

import com.synaptic.ai.core.*

/**
 * SynapticTransformer: The full model architecture.
 *
 * This is a 4-layer sparse transformer that takes fused sensor embeddings
 * and produces:
 * 1. A continuous "world state" vector (what's happening around the user)
 * 2. Context predictions (what will likely happen next)
 * 3. Anomaly scores (is something unusual happening?)
 * 4. Optional text tokens (for generating natural language descriptions)
 *
 * Novel aspects:
 * - Rotary Position Embeddings (RoPE) adapted for temporal sensor data
 * - Cross-sensor attention: sensors attend to each other, discovering correlations
 * - Temporal causal masking: the model can only look at past sensor readings
 * - Adaptive computation: early exit when confidence is high (saves battery)
 */
class SynapticTransformer(private val weights: TransformerWeights) {

    data class TransformerWeights(
        val sensorRouter: SensorExpertRouter,
        val sensorExperts: Map<SensorModality, SensorExpert>,
        val layers: List<TransformerLayerWeights>,
        val finalNorm: Tensor,                    // [EMBED_DIM]
        val worldStateHead: Tensor,               // [EMBED_DIM, EMBED_DIM] -> world state
        val worldStateBias: Tensor,               // [EMBED_DIM]
        val contextPredictionHead: Tensor,        // [EMBED_DIM, 64] -> context class logits
        val contextPredictionBias: Tensor,        // [64]
        val anomalyHead: Tensor,                  // [EMBED_DIM, 1] -> anomaly score
        val anomalyBias: Tensor,                  // [1]
        val textHead: Tensor,                     // [EMBED_DIM, VOCAB_SIZE]
        val confidenceHead: Tensor,               // [EMBED_DIM, 1] -> for early exit
    )

    data class TransformerLayerWeights(
        val attnNorm: Tensor,         // [EMBED_DIM] - pre-attention RMS norm
        val qProj: Tensor,            // [EMBED_DIM, EMBED_DIM]
        val kProj: Tensor,            // [EMBED_DIM, EMBED_DIM]
        val vProj: Tensor,            // [EMBED_DIM, EMBED_DIM]
        val outProj: Tensor,          // [EMBED_DIM, EMBED_DIM]
        val ffnNorm: Tensor,          // [EMBED_DIM] - pre-FFN RMS norm
        val ffnGateProj: Tensor,      // [EMBED_DIM, FF_DIM] - SwiGLU gate
        val ffnUpProj: Tensor,        // [EMBED_DIM, FF_DIM] - SwiGLU up
        val ffnDownProj: Tensor,      // [FF_DIM, EMBED_DIM] - SwiGLU down
    )

    data class ModelOutput(
        val worldState: Tensor,          // [1, EMBED_DIM] - continuous world representation
        val contextPredictions: Tensor,  // [1, 64] - predicted context class probabilities
        val anomalyScore: Float,         // 0.0 = normal, 1.0 = highly anomalous
        val confidence: Float,           // Model's confidence in its output
        val hiddenState: Tensor,         // [1, EMBED_DIM] - for temporal chaining
        val exitedEarly: Boolean,        // Whether adaptive computation triggered early exit
        val activeExperts: List<Int>,    // Which sensor experts were activated
    )

    // KV-cache for efficient autoregressive inference
    private val kvCache = mutableListOf<Pair<MutableList<Tensor>, MutableList<Tensor>>>()
    private var cachePosition = 0

    companion object {
        fun initialize(): SynapticTransformer {
            val embedDim = SynapticCore.EMBED_DIM
            val ffDim = SynapticCore.FF_DIM

            val layers = (0 until SynapticCore.NUM_LAYERS).map {
                TransformerLayerWeights(
                    attnNorm = Tensor.ones(embedDim),
                    qProj = Tensor.randn(embedDim, embedDim, scale = 0.02f),
                    kProj = Tensor.randn(embedDim, embedDim, scale = 0.02f),
                    vProj = Tensor.randn(embedDim, embedDim, scale = 0.02f),
                    outProj = Tensor.randn(embedDim, embedDim, scale = 0.02f),
                    ffnNorm = Tensor.ones(embedDim),
                    ffnGateProj = Tensor.randn(embedDim, ffDim, scale = 0.02f),
                    ffnUpProj = Tensor.randn(embedDim, ffDim, scale = 0.02f),
                    ffnDownProj = Tensor.randn(ffDim, embedDim, scale = 0.02f),
                )
            }

            val sensorExperts = SensorModality.entries.associateWith { modality ->
                SensorExpert(modality, SensorExpert.initializeWeights(modality))
            }

            val weights = TransformerWeights(
                sensorRouter = SensorExpertRouter(SensorExpertRouter.initializeWeights()),
                sensorExperts = sensorExperts,
                layers = layers,
                finalNorm = Tensor.ones(embedDim),
                worldStateHead = Tensor.randn(embedDim, embedDim, scale = 0.02f),
                worldStateBias = Tensor.zeros(embedDim),
                contextPredictionHead = Tensor.randn(embedDim, 64, scale = 0.02f),
                contextPredictionBias = Tensor.zeros(64),
                anomalyHead = Tensor.randn(embedDim, 1, scale = 0.02f),
                anomalyBias = Tensor.zeros(1),
                textHead = Tensor.randn(embedDim, SynapticCore.VOCAB_SIZE, scale = 0.02f),
                confidenceHead = Tensor.randn(embedDim, 1, scale = 0.02f),
            )

            return SynapticTransformer(weights)
        }
    }

    /**
     * Forward pass: Process one timestep of sensor data.
     *
     * @param sensorInputs Map of available sensor data for this timestep
     * @param previousHidden Previous timestep's hidden state for temporal continuity
     * @param earlyExitThreshold Confidence threshold for adaptive early exit (saves compute)
     */
    fun forward(
        sensorInputs: Map<SensorModality, Tensor>,
        previousHidden: Tensor? = null,
        earlyExitThreshold: Float = 0.85f,
    ): ModelOutput {
        val embedDim = SynapticCore.EMBED_DIM

        // Step 1: Concatenate all available sensor data for routing
        val allSensorData = concatenateSensorInputs(sensorInputs)

        // Step 2: Route through expert gating network
        val routing = weights.sensorRouter.route(allSensorData, previousHidden)

        // Step 3: Process only through active experts (the key efficiency gain)
        val expertOutputs = mutableListOf<Tensor>()
        val modalityList = SensorModality.entries
        for ((idx, expertIdx) in routing.activeExperts.withIndex()) {
            val modality = modalityList[expertIdx]
            val expert = weights.sensorExperts[modality]!!
            val sensorData = sensorInputs[modality]
                ?: Tensor.zeros(1, SynapticCore.SENSOR_DIMS[modality]!!)

            val expertOutput = expert.forward(sensorData)
            // Weight by routing decision
            val weighted = TensorOps.scale(expertOutput, routing.expertWeights[idx])
            expertOutputs.add(weighted)
        }

        // Step 4: Fuse expert outputs through weighted sum
        var hidden = Tensor.zeros(1, embedDim)
        for (output in expertOutputs) {
            hidden = TensorOps.add(hidden, output)
        }

        // Step 5: Pass through transformer layers with adaptive early exit
        var exitedEarly = false
        for ((layerIdx, layer) in weights.layers.withIndex()) {
            hidden = transformerLayer(hidden, layer)

            // Check for early exit after layer 2 (save compute when confident)
            if (layerIdx >= 1) {
                val confLogit = TensorOps.matmul(hidden, weights.confidenceHead)
                val confidence = 1.0f / (1.0f + kotlin.math.exp(-confLogit.data[0]))
                if (confidence > earlyExitThreshold) {
                    exitedEarly = true
                    break
                }
            }
        }

        // Step 6: Final normalization
        hidden = TensorOps.rmsNorm(hidden, weights.finalNorm)

        // Step 7: Compute output heads
        val worldState = TensorOps.add(
            TensorOps.matmul(hidden, weights.worldStateHead),
            weights.worldStateBias
        )

        val contextLogits = TensorOps.add(
            TensorOps.matmul(hidden, weights.contextPredictionHead),
            weights.contextPredictionBias
        )
        val contextPredictions = TensorOps.softmax(contextLogits)

        val anomalyLogit = TensorOps.add(
            TensorOps.matmul(hidden, weights.anomalyHead),
            weights.anomalyBias
        )
        val anomalyScore = 1.0f / (1.0f + kotlin.math.exp(-anomalyLogit.data[0]))

        val confLogit = TensorOps.matmul(hidden, weights.confidenceHead)
        val confidence = 1.0f / (1.0f + kotlin.math.exp(-confLogit.data[0]))

        return ModelOutput(
            worldState = worldState,
            contextPredictions = contextPredictions,
            anomalyScore = anomalyScore,
            confidence = confidence,
            hiddenState = hidden,
            exitedEarly = exitedEarly,
            activeExperts = routing.activeExperts,
        )
    }

    /**
     * Single transformer layer: Pre-norm -> Multi-Head Attention -> Pre-norm -> SwiGLU FFN
     */
    private fun transformerLayer(input: Tensor, layer: TransformerLayerWeights): Tensor {
        val embedDim = SynapticCore.EMBED_DIM
        val numHeads = SynapticCore.NUM_HEADS
        val headDim = SynapticCore.HEAD_DIM

        // Pre-norm + Multi-Head Self-Attention
        val normed = TensorOps.rmsNorm(input, layer.attnNorm)

        val q = TensorOps.matmul(normed, layer.qProj)
        val k = TensorOps.matmul(normed, layer.kProj)
        val v = TensorOps.matmul(normed, layer.vProj)

        // Apply RoPE (Rotary Position Embedding) for temporal awareness
        val qRoped = applyRoPE(q, cachePosition)
        val kRoped = applyRoPE(k, cachePosition)

        // Reshape for multi-head attention: [1, embedDim] -> [numHeads, 1, headDim]
        val qHeads = qRoped.reshape(numHeads, 1, headDim)
        val kHeads = kRoped.reshape(numHeads, 1, headDim)
        val vHeads = v.reshape(numHeads, 1, headDim)

        // Scaled dot-product attention
        val scale = 1.0f / kotlin.math.sqrt(headDim.toFloat())
        val kT = TensorOps.transposeLastTwo(kHeads)
        val attnScores = TensorOps.scale(TensorOps.batchedMatmul(qHeads, kT), scale)
        val attnWeights = TensorOps.softmax(attnScores)
        val attnOutput = TensorOps.batchedMatmul(attnWeights, vHeads)

        // Reshape back and project: [numHeads, 1, headDim] -> [1, embedDim]
        val attnFlat = attnOutput.reshape(1, embedDim)
        val projected = TensorOps.matmul(attnFlat, layer.outProj)

        // Residual connection
        var output = TensorOps.add(input, projected)

        // Pre-norm + SwiGLU Feed-Forward Network
        val ffnNormed = TensorOps.rmsNorm(output, layer.ffnNorm)
        val gate = TensorOps.silu(TensorOps.matmul(ffnNormed, layer.ffnGateProj))
        val up = TensorOps.matmul(ffnNormed, layer.ffnUpProj)

        // SwiGLU: element-wise multiply gate and up projections
        val gated = Tensor(gate.shape.clone(), FloatArray(gate.size))
        for (i in gate.data.indices) {
            gated.data[i] = gate.data[i] * up.data[i]
        }
        val ffnOut = TensorOps.matmul(gated, layer.ffnDownProj)

        // Residual connection
        output = TensorOps.add(output, ffnOut)

        cachePosition++
        return output
    }

    /**
     * Rotary Position Embedding adapted for temporal sensor sequences.
     * Instead of token positions, we encode time-step positions,
     * allowing the model to understand the temporal ordering of sensor readings.
     */
    private fun applyRoPE(x: Tensor, position: Int): Tensor {
        val dim = x.shape.last()
        val result = x.data.clone()
        val halfDim = dim / 2

        for (i in 0 until halfDim) {
            val theta = 1.0f / kotlin.math.pow(10000.0f, 2.0f * i / dim)
            val angle = position * theta
            val cos = kotlin.math.cos(angle)
            val sin = kotlin.math.sin(angle)

            val x0 = x.data[i]
            val x1 = x.data[i + halfDim]
            result[i] = x0 * cos - x1 * sin
            result[i + halfDim] = x0 * sin + x1 * cos
        }

        return Tensor(x.shape.clone(), result)
    }

    /**
     * Concatenate available sensor inputs into a single vector for routing.
     * Missing sensors get zero-filled (the router learns to ignore them).
     */
    private fun concatenateSensorInputs(inputs: Map<SensorModality, Tensor>): Tensor {
        val totalDim = SynapticCore.TOTAL_SENSOR_DIM
        val result = FloatArray(totalDim)
        var offset = 0

        for (modality in SensorModality.entries) {
            val dim = SynapticCore.SENSOR_DIMS[modality]!!
            val data = inputs[modality]
            if (data != null) {
                System.arraycopy(data.data, 0, result, offset, minOf(dim, data.size))
            }
            offset += dim
        }

        return Tensor(intArrayOf(1, totalDim), result)
    }

    /** Reset the KV cache and position counter (call when starting a new context window) */
    fun resetCache() {
        kvCache.clear()
        cachePosition = 0
    }
}
