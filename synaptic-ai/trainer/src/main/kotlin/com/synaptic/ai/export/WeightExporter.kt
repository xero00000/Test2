package com.synaptic.ai.export

import com.synaptic.ai.training.TrainingEngine
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * WeightExporter: Serializes trained model weights to a binary format
 * that the Android app can load directly into the SynapticTransformer.
 *
 * File format (.synaptic):
 * - Magic bytes: "SYNP" (4 bytes)
 * - Version: uint32 (4 bytes)
 * - Num parameter groups: uint32 (4 bytes)
 * - For each parameter group:
 *   - Name length: uint16 (2 bytes)
 *   - Name: UTF-8 string
 *   - Num dimensions: uint8 (1 byte)
 *   - Dimensions: uint32[] (4 bytes each)
 *   - Data: float32[] (4 bytes each, little-endian)
 */
class WeightExporter {

    companion object {
        const val MAGIC = "SYNP"
        const val VERSION = 1
    }

    data class NamedParameter(
        val name: String,
        val shape: IntArray,
        val data: FloatArray,
    )

    /**
     * Export trained model weights to a .synaptic file.
     */
    fun export(model: TrainingEngine.TrainableModel, outputPath: String) {
        val params = modelToNamedParams(model)

        DataOutputStream(BufferedOutputStream(FileOutputStream(outputPath))).use { dos ->
            // Magic bytes
            dos.writeBytes(MAGIC)

            // Version
            dos.writeInt(VERSION)

            // Number of parameter groups
            dos.writeInt(params.size)

            var totalBytes = 12L  // header

            for (param in params) {
                // Name
                val nameBytes = param.name.toByteArray(Charsets.UTF_8)
                dos.writeShort(nameBytes.size)
                dos.write(nameBytes)

                // Shape
                dos.writeByte(param.shape.size)
                for (dim in param.shape) {
                    dos.writeInt(dim)
                }

                // Data (as little-endian floats for ARM compatibility)
                val buf = ByteBuffer.allocate(param.data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                for (v in param.data) {
                    buf.putFloat(v)
                }
                dos.write(buf.array())

                totalBytes += 2 + nameBytes.size + 1 + param.shape.size * 4 + param.data.size * 4
            }

            println("Exported ${params.size} parameter groups (${totalBytes / 1024}KB) to $outputPath")
        }
    }

    /**
     * Load weights from a .synaptic file.
     */
    fun load(inputPath: String): List<NamedParameter> {
        val params = mutableListOf<NamedParameter>()

        DataInputStream(BufferedInputStream(FileInputStream(inputPath))).use { dis ->
            // Verify magic
            val magic = ByteArray(4)
            dis.readFully(magic)
            require(String(magic) == MAGIC) { "Invalid file format: expected $MAGIC" }

            // Version
            val version = dis.readInt()
            require(version == VERSION) { "Unsupported version: $version" }

            // Parameter groups
            val numGroups = dis.readInt()
            for (g in 0 until numGroups) {
                // Name
                val nameLen = dis.readUnsignedShort()
                val nameBytes = ByteArray(nameLen)
                dis.readFully(nameBytes)
                val name = String(nameBytes, Charsets.UTF_8)

                // Shape
                val numDims = dis.readUnsignedByte()
                val shape = IntArray(numDims) { dis.readInt() }

                // Data
                val numFloats = shape.fold(1) { acc, d -> acc * d }
                val dataBytes = ByteArray(numFloats * 4)
                dis.readFully(dataBytes)
                val buf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
                val data = FloatArray(numFloats) { buf.float }

                params.add(NamedParameter(name, shape, data))
            }
        }

        return params
    }

    /**
     * Convert the training model's parameter arrays into named parameters
     * that map to the full SynapticTransformer architecture.
     */
    private fun modelToNamedParams(model: TrainingEngine.TrainableModel): List<NamedParameter> {
        val params = mutableListOf<NamedParameter>()

        val sensorNames = arrayOf("motion", "spatial", "visual", "audio", "ambient", "radio")
        val sensorDims = TrainingEngine.TrainableModel.SENSOR_DIMS
        val expertDim = TrainingEngine.TrainableModel.EXPERT_DIM

        // Sensor projections
        for (i in 0 until 6) {
            params.add(NamedParameter(
                "expert.${sensorNames[i]}.projection",
                intArrayOf(sensorDims[i], expertDim),
                model.sensorProjections[i],
            ))
            params.add(NamedParameter(
                "expert.${sensorNames[i]}.bias",
                intArrayOf(expertDim),
                model.sensorBiases[i],
            ))
        }

        // Routing gate
        params.add(NamedParameter(
            "router.gate_projection",
            intArrayOf(TrainingEngine.TrainableModel.TOTAL_SENSOR_DIM, 6),
            model.routingGate,
        ))
        params.add(NamedParameter(
            "router.gate_bias",
            intArrayOf(6),
            model.routingBias,
        ))

        // Cross-attention
        params.add(NamedParameter("attention.q_proj", intArrayOf(192, 64), model.crossAttentionQ))
        params.add(NamedParameter("attention.k_proj", intArrayOf(192, 64), model.crossAttentionK))
        params.add(NamedParameter("attention.v_proj", intArrayOf(192, 64), model.crossAttentionV))
        params.add(NamedParameter("attention.out_proj", intArrayOf(64, 192), model.crossAttentionOut))

        // FFN
        params.add(NamedParameter("ffn.weight1", intArrayOf(192, 128), model.ffnWeight1))
        params.add(NamedParameter("ffn.bias1", intArrayOf(128), model.ffnBias1))
        params.add(NamedParameter("ffn.weight2", intArrayOf(128, 192), model.ffnWeight2))
        params.add(NamedParameter("ffn.bias2", intArrayOf(192), model.ffnBias2))

        // Classification head
        params.add(NamedParameter("head.context", intArrayOf(192, 16), model.contextHead))
        params.add(NamedParameter("head.context_bias", intArrayOf(16), model.contextBias))

        // Anomaly head
        params.add(NamedParameter("head.anomaly", intArrayOf(192), model.anomalyHead))
        params.add(NamedParameter("head.anomaly_bias", intArrayOf(1), model.anomalyBias))

        return params
    }

    /**
     * Print model summary.
     */
    fun printModelSummary(model: TrainingEngine.TrainableModel) {
        val params = modelToNamedParams(model)
        println("\n=== Model Weight Summary ===")
        println("${"Layer".padEnd(40)} ${"Shape".padEnd(20)} Params")
        println("-".repeat(75))
        var total = 0
        for (p in params) {
            val shapeStr = p.shape.joinToString("x")
            val count = p.data.size
            total += count
            println("${p.name.padEnd(40)} ${shapeStr.padEnd(20)} $count")
        }
        println("-".repeat(75))
        println("${"Total".padEnd(40)} ${"".padEnd(20)} $total")
        println("Estimated file size: ${total * 4 / 1024}KB")
        println()
    }
}
