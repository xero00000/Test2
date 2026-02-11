package com.synaptic.ai.core

import kotlin.math.*

/**
 * SynapticCore: A novel Sparse Mixture-of-Sensors Transformer architecture
 * designed for real-time on-device inference on the Samsung S25 Ultra.
 *
 * KEY INNOVATION: Instead of treating each sensor as a separate modality fed into
 * a standard multi-modal model, SynapticCore uses dynamic "sensor routing" where
 * a lightweight gating network decides which sensor streams are relevant at each
 * inference step, activating only the necessary expert pathways. This means:
 *
 * 1. When you're walking outside, motion + GPS + barometer experts activate
 * 2. When you're in a conversation, audio + proximity + light experts activate
 * 3. When you're photographing something, camera + gyroscope + magnetometer activate
 *
 * The model learns CROSS-SENSOR CORRELATIONS that no single-modality model can:
 * - Barometric pressure drop + accelerometer stillness = elevator ride
 * - Light sensor spike + gyroscope rotation = phone picked up from table
 * - Magnetometer anomaly + GPS drift = entering a building
 *
 * Architecture: 4-layer sparse transformer, ~15M parameters (fits in S25's NPU cache)
 * - Embedding dim: 256
 * - 8 attention heads
 * - 6 sensor expert pathways
 * - Feed-forward dim: 512
 * - Context window: 128 temporal steps (~12.8 seconds at 10Hz)
 */
object SynapticCore {
    // Architecture constants tuned for Snapdragon 8 Elite NPU
    const val EMBED_DIM = 256
    const val NUM_HEADS = 8
    const val HEAD_DIM = EMBED_DIM / NUM_HEADS  // 32
    const val FF_DIM = 512
    const val NUM_LAYERS = 4
    const val CONTEXT_LENGTH = 128
    const val NUM_SENSOR_EXPERTS = 6
    const val TOP_K_EXPERTS = 3  // Only activate 3 of 6 experts per step
    const val VOCAB_SIZE = 4096  // For text token output

    // Sensor input dimensions (raw sensor data sizes)
    val SENSOR_DIMS = mapOf(
        SensorModality.MOTION to 9,       // accel(3) + gyro(3) + gravity(3)
        SensorModality.SPATIAL to 7,       // magnetometer(3) + GPS(3) + barometer(1)
        SensorModality.VISUAL to 64,       // downsampled visual feature vector from camera
        SensorModality.AUDIO to 40,        // mel-frequency cepstral coefficients
        SensorModality.AMBIENT to 4,       // light + proximity + temperature + humidity
        SensorModality.RADIO to 8          // WiFi RSSI(3) + BT RSSI(3) + UWB(2)
    )

    val TOTAL_SENSOR_DIM = SENSOR_DIMS.values.sum()  // 132
}

enum class SensorModality {
    MOTION,   // Accelerometer, Gyroscope, Gravity
    SPATIAL,  // Magnetometer, GPS, Barometer
    VISUAL,   // Camera visual features
    AUDIO,    // Microphone audio features
    AMBIENT,  // Light, Proximity, Temperature, Humidity
    RADIO     // WiFi, Bluetooth, UWB ranging
}

/**
 * Lightweight tensor class for on-device computation.
 * Backed by FloatArray for zero-copy NPU transfer.
 * Uses row-major layout compatible with TFLite and QNN.
 */
class Tensor(val shape: IntArray, val data: FloatArray) {
    val size: Int get() = data.size

    companion object {
        fun zeros(vararg dims: Int): Tensor {
            val size = dims.fold(1) { acc, d -> acc * d }
            return Tensor(dims, FloatArray(size))
        }

        fun randn(vararg dims: Int, scale: Float = 0.02f): Tensor {
            val size = dims.fold(1) { acc, d -> acc * d }
            val data = FloatArray(size) { (Math.random().toFloat() - 0.5f) * 2 * scale }
            return Tensor(dims, data)
        }

        fun ones(vararg dims: Int): Tensor {
            val size = dims.fold(1) { acc, d -> acc * d }
            return Tensor(dims, FloatArray(size) { 1.0f })
        }
    }

    operator fun get(vararg indices: Int): Float {
        var idx = 0
        var stride = 1
        for (i in shape.indices.reversed()) {
            idx += indices[i] * stride
            stride *= shape[i]
        }
        return data[idx]
    }

    operator fun set(vararg indices: Int, value: Float) {
        var idx = 0
        var stride = 1
        for (i in shape.indices.reversed()) {
            idx += indices[i] * stride
            stride *= shape[i]
        }
        data[idx] = value
    }

    fun flatIndex(vararg indices: Int): Int {
        var idx = 0
        var stride = 1
        for (i in shape.indices.reversed()) {
            idx += indices[i] * stride
            stride *= shape[i]
        }
        return idx
    }

    fun reshape(vararg newShape: Int): Tensor {
        val newSize = newShape.fold(1) { acc, d -> acc * d }
        require(newSize == size) { "Cannot reshape ${shape.toList()} to ${newShape.toList()}" }
        return Tensor(newShape, data)
    }

    fun slice(dim: Int, index: Int): Tensor {
        val newShape = shape.toMutableList().apply { removeAt(dim) }.toIntArray()
        val outerSize = shape.take(dim).fold(1) { acc, d -> acc * d }
        val innerSize = shape.drop(dim + 1).fold(1) { acc, d -> acc * d }
        val sliceSize = innerSize
        val newData = FloatArray(size / shape[dim])

        var outIdx = 0
        for (outer in 0 until outerSize) {
            val srcStart = (outer * shape[dim] + index) * innerSize
            System.arraycopy(data, srcStart, newData, outIdx, sliceSize)
            outIdx += sliceSize
        }
        return Tensor(newShape, newData)
    }
}

/**
 * Core math operations optimized for ARM NEON via JNI fallback.
 * These are the building blocks that TFLite/QNN will accelerate on the NPU.
 */
object TensorOps {
    /** Matrix multiplication: [M, K] x [K, N] -> [M, N] */
    fun matmul(a: Tensor, b: Tensor): Tensor {
        val m = a.shape[0]
        val k = a.shape[1]
        val n = b.shape[1]
        require(b.shape[0] == k) { "Shape mismatch: ${a.shape.toList()} x ${b.shape.toList()}" }

        val result = FloatArray(m * n)
        // Tiled matmul for cache-friendly access on ARM
        val tileSize = 32
        for (ii in 0 until m step tileSize) {
            for (jj in 0 until n step tileSize) {
                for (kk in 0 until k step tileSize) {
                    val iEnd = minOf(ii + tileSize, m)
                    val jEnd = minOf(jj + tileSize, n)
                    val kEnd = minOf(kk + tileSize, k)
                    for (i in ii until iEnd) {
                        for (j in jj until jEnd) {
                            var sum = result[i * n + j]
                            for (l in kk until kEnd) {
                                sum += a.data[i * k + l] * b.data[l * n + j]
                            }
                            result[i * n + j] = sum
                        }
                    }
                }
            }
        }
        return Tensor(intArrayOf(m, n), result)
    }

    /** Batched matrix multiply for multi-head attention */
    fun batchedMatmul(a: Tensor, b: Tensor): Tensor {
        // a: [batch, M, K], b: [batch, K, N]
        val batch = a.shape[0]
        val m = a.shape[1]
        val k = a.shape[2]
        val n = b.shape[2]
        val result = Tensor.zeros(batch, m, n)

        for (bi in 0 until batch) {
            val aOffset = bi * m * k
            val bOffset = bi * k * n
            val rOffset = bi * m * n
            for (i in 0 until m) {
                for (j in 0 until n) {
                    var sum = 0f
                    for (l in 0 until k) {
                        sum += a.data[aOffset + i * k + l] * b.data[bOffset + l * n + j]
                    }
                    result.data[rOffset + i * n + j] = sum
                }
            }
        }
        return result
    }

    /** Element-wise addition with broadcasting */
    fun add(a: Tensor, b: Tensor): Tensor {
        val result = Tensor(a.shape.clone(), a.data.clone())
        if (b.size == a.shape.last()) {
            // Broadcast last dimension
            val innerDim = a.shape.last()
            for (i in a.data.indices) {
                result.data[i] += b.data[i % innerDim]
            }
        } else {
            for (i in a.data.indices) {
                result.data[i] += b.data[i]
            }
        }
        return result
    }

    /** RMS Layer Normalization (more efficient than LayerNorm, used in LLaMA) */
    fun rmsNorm(x: Tensor, weight: Tensor, eps: Float = 1e-6f): Tensor {
        val dim = x.shape.last()
        val outer = x.size / dim
        val result = Tensor(x.shape.clone(), FloatArray(x.size))

        for (i in 0 until outer) {
            val offset = i * dim
            var sumSq = 0f
            for (j in 0 until dim) {
                sumSq += x.data[offset + j] * x.data[offset + j]
            }
            val rms = sqrt(sumSq / dim + eps)
            for (j in 0 until dim) {
                result.data[offset + j] = x.data[offset + j] / rms * weight.data[j]
            }
        }
        return result
    }

    /** SiLU activation (Swish) - used in modern transformers */
    fun silu(x: Tensor): Tensor {
        val result = Tensor(x.shape.clone(), FloatArray(x.size))
        for (i in x.data.indices) {
            val sigmoid = 1.0f / (1.0f + exp(-x.data[i]))
            result.data[i] = x.data[i] * sigmoid
        }
        return result
    }

    /** Softmax along last dimension */
    fun softmax(x: Tensor): Tensor {
        val dim = x.shape.last()
        val outer = x.size / dim
        val result = Tensor(x.shape.clone(), FloatArray(x.size))

        for (i in 0 until outer) {
            val offset = i * dim
            var maxVal = Float.NEGATIVE_INFINITY
            for (j in 0 until dim) {
                maxVal = maxOf(maxVal, x.data[offset + j])
            }
            var sumExp = 0f
            for (j in 0 until dim) {
                val e = exp(x.data[offset + j] - maxVal)
                result.data[offset + j] = e
                sumExp += e
            }
            for (j in 0 until dim) {
                result.data[offset + j] /= sumExp
            }
        }
        return result
    }

    /** GELU activation for gating networks */
    fun gelu(x: Tensor): Tensor {
        val result = Tensor(x.shape.clone(), FloatArray(x.size))
        for (i in x.data.indices) {
            val v = x.data[i]
            result.data[i] = 0.5f * v * (1.0f + tanh(sqrt(2.0f / PI.toFloat()) * (v + 0.044715f * v * v * v)))
        }
        return result
    }

    /** Top-K selection for expert routing */
    fun topK(scores: FloatArray, k: Int): List<Pair<Int, Float>> {
        return scores.mapIndexed { i, v -> i to v }
            .sortedByDescending { it.second }
            .take(k)
    }

    /** Transpose last two dimensions */
    fun transposeLastTwo(x: Tensor): Tensor {
        val batch = if (x.shape.size == 3) x.shape[0] else 1
        val m = x.shape[x.shape.size - 2]
        val n = x.shape[x.shape.size - 1]
        val result = Tensor(
            if (x.shape.size == 3) intArrayOf(batch, n, m) else intArrayOf(n, m),
            FloatArray(x.size)
        )
        for (b in 0 until batch) {
            val offset = b * m * n
            for (i in 0 until m) {
                for (j in 0 until n) {
                    result.data[offset + j * m + i] = x.data[offset + i * n + j]
                }
            }
        }
        return result
    }

    /** Scale tensor by scalar */
    fun scale(x: Tensor, s: Float): Tensor {
        val result = Tensor(x.shape.clone(), FloatArray(x.size))
        for (i in x.data.indices) {
            result.data[i] = x.data[i] * s
        }
        return result
    }

    /** Concatenate tensors along last dimension */
    fun concat(tensors: List<Tensor>): Tensor {
        if (tensors.size == 1) return tensors[0]
        val totalDim = tensors.sumOf { it.shape.last() }
        val outer = tensors[0].size / tensors[0].shape.last()
        val result = Tensor(
            intArrayOf(*tensors[0].shape.dropLast(1).toIntArray(), totalDim),
            FloatArray(outer * totalDim)
        )
        for (i in 0 until outer) {
            var offset = 0
            for (t in tensors) {
                val dim = t.shape.last()
                System.arraycopy(t.data, i * dim, result.data, i * totalDim + offset, dim)
                offset += dim
            }
        }
        return result
    }
}
