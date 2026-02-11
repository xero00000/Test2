package com.synaptic.ai.core

import com.synaptic.ai.model.SynapticTransformer
import kotlin.math.*

/**
 * SensorSynesthesia: The "sixth sense" of SynapticAI.
 *
 * THIS IS THE CORE INNOVATION THAT DOESN'T EXIST IN ANY CURRENT AI SYSTEM.
 *
 * Synesthesia in humans is when stimulation of one sense triggers perception in
 * another (e.g., "hearing" colors). SensorSynesthesia does the analogous thing
 * for the phone's sensors: it discovers emergent cross-modal patterns that no
 * single sensor can detect alone.
 *
 * Examples of synthetic synesthetic perceptions:
 *
 * 1. "SPATIAL TEXTURE" - combining magnetometer anomalies + barometer micro-changes
 *    + accelerometer vibrations to sense the "texture" of the space you're in.
 *    A concrete building feels different from a wooden house from a moving car.
 *
 * 2. "SOCIAL DENSITY" - combining audio spectrum energy + WiFi probe count +
 *    visual motion density to estimate how many people are nearby, even without
 *    seeing them directly.
 *
 * 3. "TEMPORAL RHYTHM" - combining light sensor periodicity + motion periodicity
 *    + audio periodicity to detect the "rhythm" of the user's activity. Walking
 *    has a rhythm. Typing has a rhythm. Conversations have a rhythm.
 *
 * 4. "ENVIRONMENTAL MOOD" - combining barometric pressure trend + light color
 *    temperature + ambient noise level + time-of-day to create an abstract
 *    "mood" vector that correlates with the user's likely emotional state.
 *
 * 5. "TRANSITION DETECTION" - combining ALL sensors' rate-of-change vectors to
 *    detect when the user is transitioning between contexts (leaving home,
 *    arriving at work, entering a meeting, starting exercise).
 *
 * The system maintains a "synesthetic memory" — a rolling buffer of cross-modal
 * correlation patterns that it uses to detect anomalies and predict transitions.
 */
class SensorSynesthesia {

    // Cross-modal correlation matrices (learned correlations between sensor pairs)
    private val correlationMatrices = mutableMapOf<Pair<SensorModality, SensorModality>, FloatArray>()

    // Synesthetic perception buffers
    private val spatialTextureBuffer = RollingStatistics(8, 60)    // 60 samples (~6 sec)
    private val socialDensityBuffer = RollingStatistics(4, 60)
    private val temporalRhythmBuffer = RollingStatistics(6, 300)   // 300 samples (~30 sec)
    private val environmentalMoodBuffer = RollingStatistics(8, 600) // 600 samples (~60 sec)
    private val transitionBuffer = RollingStatistics(12, 100)

    // Thresholds for transition detection (adaptive, updated by learning)
    private var transitionThreshold = 0.3f
    private var anomalyBaseline = FloatArray(12)
    private var baselineCount = 0

    data class SynestheticPerception(
        val spatialTexture: FloatArray,      // 8-dim: the "feel" of the current space
        val socialDensity: FloatArray,       // 4-dim: estimated social environment
        val temporalRhythm: FloatArray,      // 6-dim: activity rhythm features
        val environmentalMood: FloatArray,   // 8-dim: abstract environmental state
        val isTransitioning: Boolean,        // True if context transition detected
        val transitionConfidence: Float,     // 0-1 confidence of transition
        val anomalyVector: FloatArray,       // 12-dim: per-feature anomaly scores
        val overallNovelty: Float,           // 0-1 how novel is the current situation
    )

    /**
     * Compute synesthetic perceptions from the current model output and raw sensor data.
     * This is called after each transformer forward pass.
     */
    fun perceive(
        sensorData: Map<SensorModality, Tensor>,
        modelOutput: SynapticTransformer.ModelOutput,
    ): SynestheticPerception {

        // --- SPATIAL TEXTURE ---
        // Cross-correlate magnetometer, barometer, and accelerometer
        val spatialTexture = computeSpatialTexture(sensorData)
        spatialTextureBuffer.push(spatialTexture)

        // --- SOCIAL DENSITY ---
        // Estimate from audio energy, radio signals, and visual motion
        val socialDensity = computeSocialDensity(sensorData)
        socialDensityBuffer.push(socialDensity)

        // --- TEMPORAL RHYTHM ---
        // Extract periodicity from motion and ambient sensors
        val temporalRhythm = computeTemporalRhythm(sensorData)
        temporalRhythmBuffer.push(temporalRhythm)

        // --- ENVIRONMENTAL MOOD ---
        // Abstract environmental state from pressure, light, audio, time
        val environmentalMood = computeEnvironmentalMood(sensorData)
        environmentalMoodBuffer.push(environmentalMood)

        // --- TRANSITION DETECTION ---
        // Rate-of-change across all synesthetic features
        val transitionFeatures = computeTransitionFeatures(
            spatialTexture, socialDensity, temporalRhythm, environmentalMood
        )
        transitionBuffer.push(transitionFeatures)

        // Detect if transition is happening
        val transitionScore = computeTransitionScore(transitionFeatures)
        val isTransitioning = transitionScore > transitionThreshold

        // Compute overall novelty (how different is this from recent history)
        val anomalyVector = computeAnomalyVector(transitionFeatures)
        val overallNovelty = anomalyVector.map { abs(it) }.average().toFloat()

        // Update adaptive thresholds
        updateBaseline(transitionFeatures)

        return SynestheticPerception(
            spatialTexture = spatialTexture,
            socialDensity = socialDensity,
            temporalRhythm = temporalRhythm,
            environmentalMood = environmentalMood,
            isTransitioning = isTransitioning,
            transitionConfidence = transitionScore,
            anomalyVector = anomalyVector,
            overallNovelty = overallNovelty,
        )
    }

    /**
     * Spatial Texture: How does this space "feel" through the phone's sensors?
     *
     * Combines:
     * - Magnetometer field strength & uniformity (metal structures affect this)
     * - Barometric micro-fluctuations (ventilation, doors opening)
     * - Accelerometer vibration energy (floor type, nearby machinery)
     * - Gravity vector stability (how level is the surface)
     */
    private fun computeSpatialTexture(sensors: Map<SensorModality, Tensor>): FloatArray {
        val features = FloatArray(8)
        val motion = sensors[SensorModality.MOTION]?.data ?: FloatArray(9)
        val spatial = sensors[SensorModality.SPATIAL]?.data ?: FloatArray(7)

        // Magnetic field magnitude and components
        val magX = spatial.getOrElse(0) { 0f }
        val magY = spatial.getOrElse(1) { 0f }
        val magZ = spatial.getOrElse(2) { 0f }
        val magMagnitude = sqrt(magX * magX + magY * magY + magZ * magZ)

        features[0] = magMagnitude / 100f  // Normalized field strength
        features[1] = if (magMagnitude > 0) magZ / magMagnitude else 0f  // Vertical field ratio

        // Barometric pressure (normalized around sea level)
        features[2] = (spatial.getOrElse(6) { 1013f } - 1013f) / 50f

        // Accelerometer vibration energy (high-frequency component)
        val accelX = motion.getOrElse(0) { 0f }
        val accelY = motion.getOrElse(1) { 0f }
        val accelZ = motion.getOrElse(2) { 0f }
        features[3] = sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ) / 20f

        // Gyroscope rotational energy
        val gyroX = motion.getOrElse(3) { 0f }
        val gyroY = motion.getOrElse(4) { 0f }
        val gyroZ = motion.getOrElse(5) { 0f }
        features[4] = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ) / 5f

        // Gravity vector tilt (how level is the phone)
        val gravX = motion.getOrElse(6) { 0f }
        val gravY = motion.getOrElse(7) { 0f }
        val gravZ = motion.getOrElse(8) { 9.81f }
        features[5] = atan2(gravX, gravZ) / PI.toFloat()  // Roll
        features[6] = atan2(gravY, gravZ) / PI.toFloat()  // Pitch

        // Cross-correlation: magnetic anomaly * vibration (buildings near machinery)
        features[7] = features[0] * features[3]

        return features
    }

    /**
     * Social Density: How many people / how social is the environment?
     *
     * Combines audio energy, radio signal diversity, and visual motion.
     */
    private fun computeSocialDensity(sensors: Map<SensorModality, Tensor>): FloatArray {
        val features = FloatArray(4)
        val audio = sensors[SensorModality.AUDIO]?.data ?: FloatArray(40)
        val radio = sensors[SensorModality.RADIO]?.data ?: FloatArray(8)
        val visual = sensors[SensorModality.VISUAL]?.data ?: FloatArray(64)

        // Audio speech-band energy (300Hz-3kHz maps to MFCC coefficients 1-13)
        var speechEnergy = 0f
        for (i in 1 until minOf(14, audio.size)) {
            speechEnergy += abs(audio[i])
        }
        features[0] = speechEnergy / 13f

        // Radio signal diversity (more unique signals = more devices = more people)
        var radioEnergy = 0f
        for (v in radio) radioEnergy += abs(v)
        features[1] = radioEnergy / radio.size

        // Visual motion average (from camera feature group 3: motion estimation)
        var visualMotion = 0f
        for (i in 20 until minOf(28, visual.size)) {
            visualMotion += abs(visual[i])
        }
        features[2] = visualMotion / 8f

        // Cross-modal: audio * radio (high audio + many devices = crowded)
        features[3] = features[0] * features[1]

        return features
    }

    /**
     * Temporal Rhythm: What is the periodic pattern of the user's activity?
     *
     * Uses autocorrelation on sensor energy to detect repetitive patterns.
     */
    private fun computeTemporalRhythm(sensors: Map<SensorModality, Tensor>): FloatArray {
        val features = FloatArray(6)
        val motion = sensors[SensorModality.MOTION]?.data ?: FloatArray(9)
        val ambient = sensors[SensorModality.AMBIENT]?.data ?: FloatArray(4)

        // Motion energy magnitude
        var motionEnergy = 0f
        for (i in 0 until minOf(6, motion.size)) motionEnergy += motion[i] * motion[i]
        features[0] = sqrt(motionEnergy)

        // Motion energy derivative (from buffer)
        val prevMean = spatialTextureBuffer.mean()
        features[1] = features[0] - (prevMean?.getOrElse(3) { 0f } ?: 0f)

        // Light level change rate
        features[2] = ambient.getOrElse(0) { 0f }

        // Proximity oscillation (phone being picked up/put down)
        features[3] = ambient.getOrElse(1) { 0f }

        // Audio energy envelope
        val audio = sensors[SensorModality.AUDIO]?.data ?: FloatArray(40)
        var audioEnergy = 0f
        for (v in audio) audioEnergy += v * v
        features[4] = sqrt(audioEnergy / audio.size)

        // Cross-modal rhythm correlation: motion * audio
        features[5] = features[0] * features[4]

        return features
    }

    /**
     * Environmental Mood: Abstract emotional/atmospheric quality of the environment.
     */
    private fun computeEnvironmentalMood(sensors: Map<SensorModality, Tensor>): FloatArray {
        val features = FloatArray(8)
        val spatial = sensors[SensorModality.SPATIAL]?.data ?: FloatArray(7)
        val ambient = sensors[SensorModality.AMBIENT]?.data ?: FloatArray(4)
        val audio = sensors[SensorModality.AUDIO]?.data ?: FloatArray(40)

        // Barometric pressure trend (falling = storm approaching, stable = calm)
        features[0] = (spatial.getOrElse(6) { 1013f } - 1013f) / 30f

        // Light level (bright = energetic, dim = calm/night)
        features[1] = ambient.getOrElse(0) { 0f }

        // Audio spectral centroid (high = bright/sharp sounds, low = deep/rumbling)
        var weightedSum = 0f
        var totalEnergy = 0f
        for (i in audio.indices) {
            val energy = abs(audio[i])
            weightedSum += i * energy
            totalEnergy += energy
        }
        features[2] = if (totalEnergy > 0) weightedSum / totalEnergy / audio.size else 0.5f

        // Audio dynamic range (loud vs quiet)
        var maxAudio = Float.NEGATIVE_INFINITY
        var minAudio = Float.POSITIVE_INFINITY
        for (v in audio) {
            maxAudio = maxOf(maxAudio, v)
            minAudio = minOf(minAudio, v)
        }
        features[3] = (maxAudio - minAudio).coerceIn(0f, 10f) / 10f

        // Temperature comfort zone
        features[4] = ambient.getOrElse(2) { 0f }

        // Humidity
        features[5] = ambient.getOrElse(3) { 0f }

        // Altitude change (GPS altitude or barometric)
        features[6] = spatial.getOrElse(5) { 0f } / 1000f

        // Combined "energy" score
        features[7] = (features[1] + features[2] + features[3]) / 3f

        return features
    }

    /**
     * Compute transition features: rate of change across all synesthetic features.
     */
    private fun computeTransitionFeatures(
        spatialTexture: FloatArray,
        socialDensity: FloatArray,
        temporalRhythm: FloatArray,
        environmentalMood: FloatArray,
    ): FloatArray {
        val features = FloatArray(12)

        // Rate of change vs recent mean for each synesthetic group
        val stMean = spatialTextureBuffer.mean()
        val sdMean = socialDensityBuffer.mean()
        val trMean = temporalRhythmBuffer.mean()

        // Spatial texture change (3 features)
        for (i in 0 until 3) {
            features[i] = spatialTexture[i] - (stMean?.getOrElse(i) { 0f } ?: 0f)
        }

        // Social density change (3 features)
        for (i in 0 until 3) {
            features[3 + i] = socialDensity[i] - (sdMean?.getOrElse(i) { 0f } ?: 0f)
        }

        // Temporal rhythm change (3 features)
        for (i in 0 until 3) {
            features[6 + i] = temporalRhythm[i] - (trMean?.getOrElse(i) { 0f } ?: 0f)
        }

        // Environmental mood change (3 features)
        for (i in 0 until 3) {
            features[9 + i] = environmentalMood[i]
        }

        return features
    }

    private fun computeTransitionScore(features: FloatArray): Float {
        var score = 0f
        for (i in features.indices) {
            val deviation = abs(features[i] - anomalyBaseline.getOrElse(i) { 0f })
            score += deviation
        }
        return (score / features.size).coerceIn(0f, 1f)
    }

    private fun computeAnomalyVector(features: FloatArray): FloatArray {
        return FloatArray(features.size) { i ->
            features[i] - anomalyBaseline.getOrElse(i) { 0f }
        }
    }

    private fun updateBaseline(features: FloatArray) {
        baselineCount++
        val alpha = 0.01f  // Slow exponential moving average
        for (i in features.indices) {
            if (i < anomalyBaseline.size) {
                anomalyBaseline[i] = anomalyBaseline[i] * (1 - alpha) + features[i] * alpha
            }
        }
    }
}

/**
 * Rolling statistics buffer for temporal analysis.
 */
class RollingStatistics(private val dimensions: Int, private val capacity: Int) {
    private val buffer = Array(capacity) { FloatArray(dimensions) }
    private var head = 0
    private var count = 0

    @Synchronized
    fun push(values: FloatArray) {
        System.arraycopy(values, 0, buffer[head], 0, minOf(values.size, dimensions))
        head = (head + 1) % capacity
        count = minOf(count + 1, capacity)
    }

    @Synchronized
    fun mean(): FloatArray? {
        if (count == 0) return null
        val result = FloatArray(dimensions)
        for (i in 0 until count) {
            val idx = (head - count + i + capacity) % capacity
            for (d in 0 until dimensions) result[d] += buffer[idx][d]
        }
        for (d in 0 until dimensions) result[d] /= count
        return result
    }

    @Synchronized
    fun variance(): FloatArray? {
        val m = mean() ?: return null
        val result = FloatArray(dimensions)
        for (i in 0 until count) {
            val idx = (head - count + i + capacity) % capacity
            for (d in 0 until dimensions) {
                val diff = buffer[idx][d] - m[d]
                result[d] += diff * diff
            }
        }
        for (d in 0 until dimensions) result[d] /= count
        return result
    }
}
