package com.synaptic.ai.training

import com.synaptic.ai.data.SyntheticDataGenerator
import kotlin.math.*
import kotlin.random.Random

/**
 * AdamW optimizer implemented from scratch.
 * Follows the original paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
 */
class AdamWOptimizer(
    private val parameters: List<FloatArray>,
    private val lr: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val eps: Float = 1e-8f,
    private val weightDecay: Float = 0.01f,
) {
    private val m: List<FloatArray> = parameters.map { FloatArray(it.size) }  // First moment
    private val v: List<FloatArray> = parameters.map { FloatArray(it.size) }  // Second moment
    private var t = 0

    fun step(gradients: List<FloatArray>) {
        t++
        val bc1 = 1f - beta1.pow(t)
        val bc2 = 1f - beta2.pow(t)

        for (p in parameters.indices) {
            val param = parameters[p]
            val grad = gradients[p]
            val mP = m[p]
            val vP = v[p]

            for (i in param.indices) {
                // Update biased first moment estimate
                mP[i] = beta1 * mP[i] + (1 - beta1) * grad[i]
                // Update biased second moment estimate
                vP[i] = beta2 * vP[i] + (1 - beta2) * grad[i] * grad[i]

                // Bias-corrected estimates
                val mHat = mP[i] / bc1
                val vHat = vP[i] / bc2

                // AdamW update: separate weight decay from adaptive learning rate
                param[i] -= lr * (mHat / (sqrt(vHat) + eps) + weightDecay * param[i])
            }
        }
    }
}

/**
 * Main trainer class: orchestrates the full training process.
 */
class Trainer {

    private val engine = TrainingEngine()
    private val rng = Random(42)

    data class TrainingConfig(
        val epochs: Int = 30,
        val batchSize: Int = 32,
        val learningRate: Float = 0.002f,
        val weightDecay: Float = 0.01f,
        val validationSplit: Float = 0.15f,
        val gradientClipNorm: Float = 1.0f,
        val samplesPerContext: Int = 150,
        val sequenceLength: Int = 16,
        val lrSchedule: Boolean = true,
    )

    data class TrainingResult(
        val model: TrainingEngine.TrainableModel,
        val finalTrainLoss: Float,
        val finalValLoss: Float,
        val finalTrainAccuracy: Float,
        val finalValAccuracy: Float,
        val epochHistory: List<EpochMetrics>,
    )

    data class EpochMetrics(
        val epoch: Int,
        val trainLoss: Float,
        val valLoss: Float,
        val trainAccuracy: Float,
        val valAccuracy: Float,
        val learningRate: Float,
        val elapsedMs: Long,
    )

    /**
     * Run the full training pipeline.
     */
    fun train(config: TrainingConfig = TrainingConfig()): TrainingResult {
        println("=== SynapticAI Training Pipeline ===")
        println("Generating synthetic sensor data...")

        // Generate dataset
        val generator = SyntheticDataGenerator(seed = 42)
        val fullDataset = generator.generateDataset(
            samplesPerContext = config.samplesPerContext,
            sequenceLength = config.sequenceLength,
        )
        println("Generated ${fullDataset.size} training samples across ${SyntheticDataGenerator.ActivityContext.entries.size} contexts")

        // Split into train/validation
        val splitIdx = (fullDataset.size * (1 - config.validationSplit)).toInt()
        val trainData = fullDataset.subList(0, splitIdx)
        val valData = fullDataset.subList(splitIdx, fullDataset.size)
        println("Train: ${trainData.size} samples, Validation: ${valData.size} samples")

        // Initialize model
        val model = TrainingEngine.TrainableModel.initialize(rng)
        println("Model initialized: ${model.totalParams()} parameters")

        // Initialize optimizer
        val optimizer = AdamWOptimizer(
            parameters = model.allParameters(),
            lr = config.learningRate,
            weightDecay = config.weightDecay,
        )

        // Pre-training evaluation
        val (preAcc, preAnomalyErr) = engine.evaluateAccuracy(model, valData.take(200))
        println("Pre-training accuracy: ${(preAcc * 100).toInt()}% (random baseline: ~6%)")
        println()

        // Training loop
        val epochHistory = mutableListOf<EpochMetrics>()
        var bestValAcc = 0f

        for (epoch in 1..config.epochs) {
            val epochStart = System.currentTimeMillis()
            var epochLoss = 0f
            var epochSamples = 0

            // Learning rate schedule: cosine annealing with warmup
            val currentLr = if (config.lrSchedule) {
                val warmupEpochs = 3
                if (epoch <= warmupEpochs) {
                    config.learningRate * epoch / warmupEpochs
                } else {
                    val progress = (epoch - warmupEpochs).toFloat() / (config.epochs - warmupEpochs)
                    config.learningRate * 0.5f * (1 + cos(PI.toFloat() * progress))
                }
            } else {
                config.learningRate
            }

            // Shuffle training data each epoch
            val shuffled = trainData.shuffled(rng)

            // Process mini-batches
            for (batchStart in 0 until shuffled.size step config.batchSize) {
                val batchEnd = minOf(batchStart + config.batchSize, shuffled.size)
                val batch = shuffled.subList(batchStart, batchEnd)

                // Accumulate gradients over batch
                val allParams = model.allParameters()
                val gradAccum = allParams.map { FloatArray(it.size) }

                for (sample in batch) {
                    // Use a representative frame from the sequence
                    val frameIdx = sample.sensorSequence.size / 2
                    val sensorSample = sample.sensorSequence[frameIdx]

                    // Compute gradients via finite differences
                    val grads = computeGradients(model, sensorSample, sample.contextLabel, sample.anomalyScore)

                    // Accumulate
                    for (p in grads.indices) {
                        for (i in grads[p].indices) {
                            gradAccum[p][i] += grads[p][i] / batch.size
                        }
                    }

                    // Track loss
                    val result = engine.forward(model, sensorSample)
                    epochLoss += engine.computeLoss(result, sample.contextLabel, sample.anomalyScore)
                    epochSamples++
                }

                // Gradient clipping
                clipGradients(gradAccum, config.gradientClipNorm)

                // Scale gradients by learning rate ratio
                val lrScale = currentLr / config.learningRate
                for (g in gradAccum) {
                    for (i in g.indices) g[i] *= lrScale
                }

                // Optimizer step
                optimizer.step(gradAccum)
            }

            // Epoch evaluation
            val avgTrainLoss = epochLoss / epochSamples
            val (trainAcc, _) = engine.evaluateAccuracy(model, trainData.shuffled(rng).take(300))
            val (valAcc, valAnomalyErr) = engine.evaluateAccuracy(model, valData)

            // Validation loss
            var valLoss = 0f
            for (sample in valData.take(200)) {
                val sensorSample = sample.sensorSequence[sample.sensorSequence.size / 2]
                val result = engine.forward(model, sensorSample)
                valLoss += engine.computeLoss(result, sample.contextLabel, sample.anomalyScore)
            }
            valLoss /= minOf(200, valData.size)

            val elapsed = System.currentTimeMillis() - epochStart

            val metrics = EpochMetrics(
                epoch = epoch,
                trainLoss = avgTrainLoss,
                valLoss = valLoss,
                trainAccuracy = trainAcc,
                valAccuracy = valAcc,
                learningRate = currentLr,
                elapsedMs = elapsed,
            )
            epochHistory.add(metrics)

            if (valAcc > bestValAcc) bestValAcc = valAcc

            // Print progress
            val bar = buildString {
                val pct = epoch * 20 / config.epochs
                append("[")
                repeat(pct) { append("=") }
                if (pct < 20) append(">")
                repeat(maxOf(0, 19 - pct)) { append(" ") }
                append("]")
            }

            println("Epoch ${"$epoch".padStart(2)}/${config.epochs} $bar " +
                "loss: ${"%.4f".format(avgTrainLoss)} | " +
                "val_loss: ${"%.4f".format(valLoss)} | " +
                "acc: ${"%.1f".format(trainAcc * 100)}% | " +
                "val_acc: ${"%.1f".format(valAcc * 100)}% | " +
                "lr: ${"%.5f".format(currentLr)} | " +
                "${elapsed}ms")
        }

        println()
        println("=== Training Complete ===")
        println("Best validation accuracy: ${"%.1f".format(bestValAcc * 100)}%")

        val finalMetrics = epochHistory.last()
        return TrainingResult(
            model = model,
            finalTrainLoss = finalMetrics.trainLoss,
            finalValLoss = finalMetrics.valLoss,
            finalTrainAccuracy = finalMetrics.trainAccuracy,
            finalValAccuracy = finalMetrics.valAccuracy,
            epochHistory = epochHistory,
        )
    }

    /**
     * Compute parameter gradients using forward-mode finite differences.
     *
     * For each parameter, we compute: dL/dp ≈ (L(p+ε) - L(p-ε)) / (2ε)
     *
     * This is expensive (O(2*num_params) forward passes per sample) but correct,
     * and tractable for our relatively small model (~50K params in training mode).
     *
     * To make it feasible, we use parameter-group-level perturbation with random
     * projection (simultaneous perturbation stochastic approximation - SPSA).
     */
    private fun computeGradients(
        model: TrainingEngine.TrainableModel,
        sample: SyntheticDataGenerator.SensorSample,
        targetContext: Int,
        targetAnomaly: Float,
    ): List<FloatArray> {
        val allParams = model.allParameters()
        val gradients = allParams.map { FloatArray(it.size) }
        val eps = 0.001f

        // SPSA: perturb all parameters simultaneously with random direction
        // Much faster than per-parameter finite differences
        val perturbations = allParams.map { param ->
            FloatArray(param.size) { if (rng.nextBoolean()) 1f else -1f }
        }

        // Apply positive perturbation
        for (p in allParams.indices) {
            for (i in allParams[p].indices) {
                allParams[p][i] += eps * perturbations[p][i]
            }
        }
        val resultPlus = engine.forward(model, sample)
        val lossPlus = engine.computeLoss(resultPlus, targetContext, targetAnomaly)

        // Apply negative perturbation (2*eps from positive)
        for (p in allParams.indices) {
            for (i in allParams[p].indices) {
                allParams[p][i] -= 2 * eps * perturbations[p][i]
            }
        }
        val resultMinus = engine.forward(model, sample)
        val lossMinus = engine.computeLoss(resultMinus, targetContext, targetAnomaly)

        // Restore original parameters
        for (p in allParams.indices) {
            for (i in allParams[p].indices) {
                allParams[p][i] += eps * perturbations[p][i]
            }
        }

        // SPSA gradient estimate
        val lossGrad = (lossPlus - lossMinus) / (2 * eps)
        for (p in allParams.indices) {
            for (i in allParams[p].indices) {
                gradients[p][i] = lossGrad / perturbations[p][i]
            }
        }

        return gradients
    }

    /**
     * Clip gradient norms to prevent exploding gradients.
     */
    private fun clipGradients(gradients: List<FloatArray>, maxNorm: Float) {
        var totalNorm = 0f
        for (g in gradients) {
            for (v in g) totalNorm += v * v
        }
        totalNorm = sqrt(totalNorm)

        if (totalNorm > maxNorm) {
            val scale = maxNorm / totalNorm
            for (g in gradients) {
                for (i in g.indices) g[i] *= scale
            }
        }
    }
}
