package com.synaptic.ai.training

import com.synaptic.ai.data.SyntheticDataGenerator
import com.synaptic.ai.export.WeightExporter
import java.io.File

/**
 * Main entry point for SynapticAI training.
 *
 * Runs the full pipeline:
 * 1. Generate synthetic sensor data for 16 activity contexts
 * 2. Train the model using SPSA gradient estimation + AdamW
 * 3. Evaluate on held-out validation set
 * 4. Export trained weights to .synaptic format
 * 5. Print detailed analysis of what the model learned
 */
fun main() {
    println()
    println("  ╔═══════════════════════════════════════════════════╗")
    println("  ║          SynapticAI Training Pipeline             ║")
    println("  ║   Sparse Mixture-of-Sensors Transformer v1.0     ║")
    println("  ║   Target: Samsung S25 Ultra (Snapdragon 8 Elite) ║")
    println("  ╚═══════════════════════════════════════════════════╝")
    println()

    // Configure training
    val config = Trainer.TrainingConfig(
        epochs = 30,
        batchSize = 16,
        learningRate = 0.003f,
        weightDecay = 0.01f,
        validationSplit = 0.15f,
        gradientClipNorm = 1.0f,
        samplesPerContext = 150,
        sequenceLength = 16,
        lrSchedule = true,
    )

    println("Training Configuration:")
    println("  Epochs:           ${config.epochs}")
    println("  Batch Size:       ${config.batchSize}")
    println("  Learning Rate:    ${config.learningRate}")
    println("  Weight Decay:     ${config.weightDecay}")
    println("  Validation Split: ${(config.validationSplit * 100).toInt()}%")
    println("  Gradient Clip:    ${config.gradientClipNorm}")
    println("  Samples/Context:  ${config.samplesPerContext}")
    println()

    // Run training
    val trainer = Trainer()
    val result = trainer.train(config)

    // Print model summary
    val exporter = WeightExporter()
    exporter.printModelSummary(result.model)

    // Export weights
    val outputDir = "trained_weights"
    File(outputDir).mkdirs()
    val weightsPath = "$outputDir/synaptic_v1.synaptic"
    exporter.export(result.model, weightsPath)

    // Verify export by reloading
    val reloaded = exporter.load(weightsPath)
    println("Verified: loaded ${reloaded.size} parameter groups from $weightsPath")

    // Detailed per-context evaluation
    println("\n=== Per-Context Accuracy ===")
    val engine = TrainingEngine()
    val generator = SyntheticDataGenerator(seed = 99)  // Different seed for test
    val testData = generator.generateDataset(samplesPerContext = 50, sequenceLength = 16)

    val contextNames = SyntheticDataGenerator.ActivityContext.entries.map { it.name }
    val perContextCorrect = IntArray(contextNames.size)
    val perContextTotal = IntArray(contextNames.size)

    for (sample in testData) {
        val sensorSample = sample.sensorSequence[sample.sensorSequence.size / 2]
        val fwdResult = engine.forward(result.model, sensorSample)
        val predicted = fwdResult.contextLogits.indices.maxByOrNull { fwdResult.contextLogits[it] } ?: 0
        perContextTotal[sample.contextLabel]++
        if (predicted == sample.contextLabel) perContextCorrect[sample.contextLabel]++
    }

    for (i in contextNames.indices) {
        if (perContextTotal[i] > 0) {
            val acc = perContextCorrect[i] * 100f / perContextTotal[i]
            val bar = "█".repeat((acc / 5).toInt()) + "░".repeat(maxOf(0, 20 - (acc / 5).toInt()))
            println("  ${contextNames[i].padEnd(20)} $bar ${"%.1f".format(acc)}%  (${perContextCorrect[i]}/${perContextTotal[i]})")
        }
    }

    // Routing analysis: which experts activate for which contexts
    println("\n=== Expert Routing Analysis ===")
    println("Shows which sensor experts the model learned to activate for each context:")
    val expertNames = arrayOf("MOTION", "SPATIAL", "VISUAL", "AUDIO", "AMBIENT", "RADIO")

    for (ctx in SyntheticDataGenerator.ActivityContext.entries) {
        if (ctx == SyntheticDataGenerator.ActivityContext.TRANSITION) continue
        val samples = testData.filter { it.contextLabel == ctx.id }.take(20)
        val avgRouting = FloatArray(6)
        for (sample in samples) {
            val sensorSample = sample.sensorSequence[sample.sensorSequence.size / 2]
            val fwdResult = engine.forward(result.model, sensorSample)
            for (i in fwdResult.routingWeights.indices) {
                avgRouting[i] += fwdResult.routingWeights[i]
            }
        }
        for (i in avgRouting.indices) avgRouting[i] /= samples.size

        val routingStr = expertNames.indices.joinToString(" ") { i ->
            val pct = (avgRouting[i] * 100).toInt()
            "${expertNames[i].take(3)}:${pct}%".padEnd(10)
        }
        println("  ${ctx.name.padEnd(20)} $routingStr")
    }

    // Final summary
    println("\n=== Training Summary ===")
    println("  Final train accuracy:      ${"%.1f".format(result.finalTrainAccuracy * 100)}%")
    println("  Final validation accuracy:  ${"%.1f".format(result.finalValAccuracy * 100)}%")
    println("  Final train loss:           ${"%.4f".format(result.finalTrainLoss)}")
    println("  Final validation loss:      ${"%.4f".format(result.finalValLoss)}")
    println("  Weights exported to:        $weightsPath")
    println("  Ready for deployment on Samsung S25 Ultra")
    println()
}
