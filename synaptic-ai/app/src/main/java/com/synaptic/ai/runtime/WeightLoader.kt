package com.synaptic.ai.runtime

import android.content.Context
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Loads pre-trained weights from the .synaptic binary format in app assets.
 * These weights were trained on synthetic sensor data and encode learned
 * cross-sensor correlations for 16 activity contexts.
 */
class WeightLoader(private val context: Context) {

    data class NamedWeight(
        val name: String,
        val shape: IntArray,
        val data: FloatArray,
    )

    /**
     * Load all weights from the bundled asset file.
     * Returns a map of parameter name -> float array for easy access.
     */
    fun loadFromAssets(filename: String = "synaptic_v1.synaptic"): Map<String, NamedWeight> {
        val weights = mutableMapOf<String, NamedWeight>()

        context.assets.open(filename).use { stream ->
            val dis = DataInputStream(stream)

            // Verify magic
            val magic = ByteArray(4)
            dis.readFully(magic)
            require(String(magic) == "SYNP") { "Invalid weight file format" }

            // Version
            val version = dis.readInt()
            require(version == 1) { "Unsupported weight version: $version" }

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

                // Data (little-endian floats)
                val numFloats = shape.fold(1) { acc, d -> acc * d }
                val dataBytes = ByteArray(numFloats * 4)
                dis.readFully(dataBytes)
                val buf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
                val data = FloatArray(numFloats) { buf.float }

                weights[name] = NamedWeight(name, shape, data)
            }
        }

        return weights
    }

    /**
     * Verify that all expected weight groups are present.
     */
    fun verify(weights: Map<String, NamedWeight>): Boolean {
        val expected = listOf(
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
            "head.anomaly", "head.anomaly_bias",
        )
        return expected.all { it in weights }
    }
}
