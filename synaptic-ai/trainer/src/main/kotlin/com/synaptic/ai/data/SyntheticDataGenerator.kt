package com.synaptic.ai.data

import kotlin.math.*
import kotlin.random.Random

/**
 * SyntheticDataGenerator: Creates realistic simulated sensor data for training.
 *
 * Generates labeled multi-sensor sequences for 16 activity contexts that the
 * SynapticAI model needs to learn. Each context has characteristic sensor
 * patterns derived from real-world physics:
 *
 * - Walking: periodic accelerometer, changing GPS, stable barometer
 * - Sitting at desk: minimal motion, stable everything, keyboard typing audio
 * - Driving: vibration pattern, fast GPS changes, road audio
 * - In a meeting: speech audio, many radio signals, minimal motion
 * - Sleeping: near-zero motion, dark, quiet
 * - Cooking: standing motion, heat, kitchen sounds
 * - Exercising: high motion energy, heart rate up
 * - Elevator: barometer change, no GPS change, minimal horizontal motion
 * - On phone call: speech audio, phone near face (proximity)
 * - Outdoors nature: wind audio, bright light, GPS movement
 * - Public transit: vibration + GPS movement + crowd audio
 * - Shopping: walking pattern + many radio signals + varied visual
 * - Stairs: accelerometer step pattern + barometer change
 * - Cycling: rhythmic motion + fast GPS + wind audio
 * - Idle/table: zero motion, ambient light, ambient sound
 * - Transition: changing patterns between two contexts
 */
class SyntheticDataGenerator(private val seed: Long = 42L) {

    private val rng = Random(seed)

    // Context labels
    enum class ActivityContext(val id: Int) {
        WALKING(0),
        SITTING_DESK(1),
        DRIVING(2),
        IN_MEETING(3),
        SLEEPING(4),
        COOKING(5),
        EXERCISING(6),
        ELEVATOR(7),
        PHONE_CALL(8),
        OUTDOORS_NATURE(9),
        PUBLIC_TRANSIT(10),
        SHOPPING(11),
        CLIMBING_STAIRS(12),
        CYCLING(13),
        IDLE_TABLE(14),
        TRANSITION(15),
    }

    data class SensorSample(
        val motion: FloatArray,      // 9: accel(3) + gyro(3) + gravity(3)
        val spatial: FloatArray,     // 7: mag(3) + gps(3) + baro(1)
        val visual: FloatArray,      // 64: camera features
        val audio: FloatArray,       // 40: MFCCs
        val ambient: FloatArray,     // 4: light + proximity + temp + humidity
        val radio: FloatArray,       // 8: wifi(3) + bt(3) + uwb(2)
    )

    data class TrainingSample(
        val sensorSequence: List<SensorSample>,    // Temporal sequence of sensor readings
        val contextLabel: Int,                      // Ground truth activity context
        val isTransitioning: Boolean,               // Is this a transition between contexts?
        val anomalyScore: Float,                    // 0 = normal, 1 = anomalous
    )

    /**
     * Generate a full training dataset.
     * @param samplesPerContext How many sequences to generate per activity context
     * @param sequenceLength Number of timesteps per sequence (at 10Hz)
     */
    fun generateDataset(
        samplesPerContext: Int = 200,
        sequenceLength: Int = 32,
    ): List<TrainingSample> {
        val dataset = mutableListOf<TrainingSample>()

        // Generate normal samples for each context
        for (context in ActivityContext.entries) {
            if (context == ActivityContext.TRANSITION) continue // Handle separately
            for (i in 0 until samplesPerContext) {
                val sequence = generateSequence(context, sequenceLength)
                dataset.add(TrainingSample(
                    sensorSequence = sequence,
                    contextLabel = context.id,
                    isTransitioning = false,
                    anomalyScore = 0f,
                ))
            }
        }

        // Generate transition samples (context changes mid-sequence)
        val contexts = ActivityContext.entries.filter { it != ActivityContext.TRANSITION }
        for (i in 0 until samplesPerContext * 2) {
            val fromCtx = contexts.random(rng)
            val toCtx = contexts.filter { it != fromCtx }.random(rng)
            val transitionPoint = rng.nextInt(sequenceLength / 4, 3 * sequenceLength / 4)

            val sequence = mutableListOf<SensorSample>()
            for (t in 0 until sequenceLength) {
                val blend = if (t < transitionPoint) 0f
                else ((t - transitionPoint).toFloat() / (sequenceLength - transitionPoint)).coerceIn(0f, 1f)

                val fromSample = generateSingleSample(fromCtx, t.toFloat() / sequenceLength)
                val toSample = generateSingleSample(toCtx, t.toFloat() / sequenceLength)
                sequence.add(blendSamples(fromSample, toSample, blend))
            }

            dataset.add(TrainingSample(
                sensorSequence = sequence,
                contextLabel = ActivityContext.TRANSITION.id,
                isTransitioning = true,
                anomalyScore = 0f,
            ))
        }

        // Generate anomaly samples (unusual sensor patterns)
        for (i in 0 until samplesPerContext) {
            val baseCtx = contexts.random(rng)
            val sequence = generateAnomalousSequence(baseCtx, sequenceLength)
            dataset.add(TrainingSample(
                sensorSequence = sequence,
                contextLabel = baseCtx.id,
                isTransitioning = false,
                anomalyScore = rng.nextFloat() * 0.5f + 0.5f,  // 0.5 to 1.0
            ))
        }

        return dataset.shuffled(rng)
    }

    private fun generateSequence(context: ActivityContext, length: Int): List<SensorSample> {
        return (0 until length).map { t ->
            generateSingleSample(context, t.toFloat() / length)
        }
    }

    /**
     * Generate a single sensor reading for a given activity context.
     * Physics-based simulation of what each sensor would read.
     */
    fun generateSingleSample(context: ActivityContext, timePhase: Float): SensorSample {
        val t = timePhase * 2 * PI.toFloat()
        val noise = { scale: Float -> (rng.nextFloat() - 0.5f) * 2 * scale }

        return when (context) {
            ActivityContext.WALKING -> SensorSample(
                motion = floatArrayOf(
                    // Accelerometer: periodic pattern from footsteps (~2Hz)
                    sin(t * 4) * 2f + noise(0.3f),           // x: lateral sway
                    cos(t * 4) * 1.5f + noise(0.2f),         // y: forward/back
                    9.81f + sin(t * 8) * 1.0f + noise(0.2f), // z: vertical bounce (double freq)
                    // Gyroscope: subtle rotation from walking
                    sin(t * 4) * 0.3f + noise(0.05f),
                    noise(0.1f),
                    sin(t * 4) * 0.2f + noise(0.05f),
                    // Gravity
                    noise(0.1f), noise(0.1f), 9.81f,
                ),
                spatial = floatArrayOf(
                    25f + noise(2f), -10f + noise(2f), -45f + noise(2f),  // Magnetometer (outdoor)
                    37.7749f + timePhase * 0.001f, -122.4194f + timePhase * 0.001f, 10f, // GPS moving
                    1013.25f + noise(0.1f),  // Barometer stable
                ),
                visual = generateVisualFeatures(brightness = 0.7f, motion = 0.4f, edges = 0.5f),
                audio = generateAudioFeatures(type = "footsteps_outdoor", energy = 0.3f),
                ambient = floatArrayOf(
                    0.6f + noise(0.1f),   // Light: outdoor daylight
                    1.0f,                  // Proximity: far (in pocket or hand)
                    22f + noise(1f),       // Temperature
                    50f + noise(5f),       // Humidity
                ),
                radio = floatArrayOf(
                    -70f + noise(5f), -80f + noise(5f), -85f + noise(5f),  // WiFi (weak, changing)
                    -90f + noise(5f), -95f + noise(5f), -88f + noise(5f),  // BT
                    noise(0.5f), noise(0.5f),  // UWB
                ),
            )

            ActivityContext.SITTING_DESK -> SensorSample(
                motion = floatArrayOf(
                    noise(0.05f), noise(0.05f), 9.81f + noise(0.02f),  // Accel: near zero
                    noise(0.01f), noise(0.01f), noise(0.01f),          // Gyro: near zero
                    0f, 0f, 9.81f,                                      // Gravity: straight down
                ),
                spatial = floatArrayOf(
                    30f + noise(0.5f), -15f + noise(0.5f), -50f + noise(0.5f), // Mag: indoor
                    37.7749f, -122.4194f, 15f,                                   // GPS: stationary
                    1013.5f + noise(0.05f),                                       // Baro: stable indoor
                ),
                visual = generateVisualFeatures(brightness = 0.4f, motion = 0.05f, edges = 0.6f),
                audio = generateAudioFeatures(type = "office", energy = 0.15f),
                ambient = floatArrayOf(
                    0.35f + noise(0.02f),  // Light: indoor office
                    0.3f + noise(0.05f),   // Proximity: near (face/desk)
                    23f + noise(0.5f),     // Temperature: AC
                    40f + noise(3f),       // Humidity: indoor
                ),
                radio = floatArrayOf(
                    -40f + noise(2f), -45f + noise(2f), -50f + noise(2f),  // WiFi: strong indoor
                    -60f + noise(3f), -65f + noise(3f), -70f + noise(3f),  // BT: moderate
                    0.5f + noise(0.1f), 0.3f + noise(0.1f),                // UWB
                ),
            )

            ActivityContext.DRIVING -> SensorSample(
                motion = floatArrayOf(
                    noise(0.3f) + sin(t * 0.5f) * 0.5f,   // Accel: turns
                    noise(0.2f) + cos(t * 0.3f) * 0.3f,   // Accel: accel/brake
                    9.81f + noise(0.15f),                    // Accel: road bumps
                    noise(0.05f), sin(t * 0.5f) * 0.1f, noise(0.02f), // Gyro: turning
                    sin(t * 0.5f) * 0.3f, 0f, 9.81f,       // Gravity: tilting in turns
                ),
                spatial = floatArrayOf(
                    20f + noise(5f), -20f + noise(5f), -40f + noise(5f), // Mag: varying (engine)
                    37.7749f + timePhase * 0.01f, -122.4194f + timePhase * 0.008f, 10f, // GPS: fast
                    1013.25f + noise(0.3f),
                ),
                visual = generateVisualFeatures(brightness = 0.5f, motion = 0.6f, edges = 0.4f),
                audio = generateAudioFeatures(type = "driving", energy = 0.4f),
                ambient = floatArrayOf(
                    0.5f + noise(0.15f),  // Light: variable (tunnels, shade)
                    0.2f + noise(0.1f),   // Proximity: in mount
                    24f + noise(2f),      // Temperature
                    45f + noise(5f),      // Humidity
                ),
                radio = floatArrayOf(
                    -75f + noise(10f), -80f + noise(10f), -85f + noise(10f), // WiFi: weak/changing
                    -50f + noise(3f), -55f + noise(3f), -60f + noise(3f),    // BT: car connection
                    noise(0.3f), noise(0.3f),
                ),
            )

            ActivityContext.IN_MEETING -> SensorSample(
                motion = floatArrayOf(
                    noise(0.08f), noise(0.08f), 9.81f + noise(0.03f),
                    noise(0.02f), noise(0.02f), noise(0.02f),
                    0f, 0f, 9.81f,
                ),
                spatial = floatArrayOf(
                    35f + noise(1f), -12f + noise(1f), -48f + noise(1f),
                    37.7749f, -122.4194f, 15f,
                    1013.4f + noise(0.05f),
                ),
                visual = generateVisualFeatures(brightness = 0.35f, motion = 0.08f, edges = 0.3f),
                audio = generateAudioFeatures(type = "speech_multiple", energy = 0.5f),
                ambient = floatArrayOf(
                    0.3f + noise(0.02f),  // Light: conference room
                    0.8f + noise(0.1f),   // Proximity: on table
                    23f + noise(0.3f),
                    42f + noise(2f),
                ),
                radio = floatArrayOf(
                    -35f + noise(2f), -40f + noise(2f), -42f + noise(2f),  // WiFi: strong
                    -55f + noise(3f), -58f + noise(3f), -60f + noise(3f),  // BT: many devices
                    0.8f + noise(0.1f), 0.6f + noise(0.1f),
                ),
            )

            ActivityContext.SLEEPING -> SensorSample(
                motion = floatArrayOf(
                    noise(0.01f), noise(0.01f), 9.81f + noise(0.005f),
                    noise(0.002f), noise(0.002f), noise(0.002f),
                    // Gravity: lying down (phone on nightstand or bed)
                    noise(0.5f), 9.5f + noise(0.3f), noise(0.5f),
                ),
                spatial = floatArrayOf(
                    28f + noise(0.2f), -18f + noise(0.2f), -52f + noise(0.2f),
                    37.7749f, -122.4194f, 10f,
                    1013.3f + noise(0.02f),
                ),
                visual = generateVisualFeatures(brightness = 0.02f, motion = 0.0f, edges = 0.01f),
                audio = generateAudioFeatures(type = "silence", energy = 0.02f),
                ambient = floatArrayOf(
                    0.01f + noise(0.005f), // Light: dark
                    1.0f,                   // Proximity: far
                    20f + noise(0.2f),     // Temperature: night
                    55f + noise(2f),
                ),
                radio = floatArrayOf(
                    -45f + noise(1f), -50f + noise(1f), -55f + noise(1f),
                    -70f + noise(2f), -75f + noise(2f), -80f + noise(2f),
                    noise(0.1f), noise(0.1f),
                ),
            )

            ActivityContext.COOKING -> SensorSample(
                motion = floatArrayOf(
                    noise(0.5f) + sin(t * 3) * 0.3f,  // Stirring, chopping
                    noise(0.4f) + cos(t * 2) * 0.4f,
                    9.81f + noise(0.3f),
                    noise(0.2f), noise(0.15f), noise(0.3f),
                    noise(0.2f), noise(0.2f), 9.81f,
                ),
                spatial = floatArrayOf(
                    32f + noise(1f), -14f + noise(1f), -46f + noise(1f),
                    37.7749f, -122.4194f, 10f,
                    1013.3f + noise(0.1f),
                ),
                visual = generateVisualFeatures(brightness = 0.5f, motion = 0.25f, edges = 0.45f),
                audio = generateAudioFeatures(type = "kitchen", energy = 0.35f),
                ambient = floatArrayOf(
                    0.45f + noise(0.05f),
                    1.0f,
                    26f + noise(2f),    // Temperature: warm from cooking
                    60f + noise(5f),    // Humidity: steam
                ),
                radio = floatArrayOf(
                    -42f + noise(2f), -48f + noise(2f), -52f + noise(2f),
                    -65f + noise(3f), -70f + noise(3f), -75f + noise(3f),
                    noise(0.2f), noise(0.2f),
                ),
            )

            ActivityContext.EXERCISING -> SensorSample(
                motion = floatArrayOf(
                    sin(t * 6) * 4f + noise(1f),     // High energy motion
                    cos(t * 6) * 3f + noise(0.8f),
                    9.81f + sin(t * 12) * 3f + noise(0.5f),
                    sin(t * 6) * 1f + noise(0.3f),
                    noise(0.5f),
                    cos(t * 6) * 0.8f + noise(0.2f),
                    noise(0.5f), noise(0.5f), 9.81f,
                ),
                spatial = floatArrayOf(
                    25f + noise(3f), -10f + noise(3f), -45f + noise(3f),
                    37.7749f + timePhase * 0.002f, -122.4194f + timePhase * 0.002f, 10f,
                    1013.2f + noise(0.15f),
                ),
                visual = generateVisualFeatures(brightness = 0.6f, motion = 0.7f, edges = 0.4f),
                audio = generateAudioFeatures(type = "exercise", energy = 0.45f),
                ambient = floatArrayOf(
                    0.55f + noise(0.1f),
                    0.5f + noise(0.3f),
                    28f + noise(2f),    // Warm
                    65f + noise(5f),    // Sweaty
                ),
                radio = floatArrayOf(
                    -65f + noise(8f), -70f + noise(8f), -75f + noise(8f),
                    -45f + noise(3f), -50f + noise(3f), -55f + noise(3f),  // BT: earbuds
                    noise(0.3f), noise(0.3f),
                ),
            )

            ActivityContext.ELEVATOR -> SensorSample(
                motion = floatArrayOf(
                    noise(0.05f), noise(0.05f),
                    9.81f + sin(t * 0.5f) * 0.8f + noise(0.05f),  // Vertical acceleration!
                    noise(0.02f), noise(0.02f), noise(0.01f),
                    0f, 0f, 9.81f + sin(t * 0.5f) * 0.5f,
                ),
                spatial = floatArrayOf(
                    40f + noise(3f), -8f + noise(3f), -55f + noise(3f),  // Mag: distorted (metal)
                    37.7749f, -122.4194f, 15f + timePhase * 20f,  // Altitude changing
                    1013.25f - timePhase * 3f,  // BAROMETER DROPPING = going up!
                ),
                visual = generateVisualFeatures(brightness = 0.25f, motion = 0.02f, edges = 0.2f),
                audio = generateAudioFeatures(type = "elevator", energy = 0.1f),
                ambient = floatArrayOf(
                    0.2f + noise(0.02f),  // Dim lighting
                    0.4f + noise(0.1f),
                    22f + noise(0.5f),
                    38f + noise(2f),
                ),
                radio = floatArrayOf(
                    -60f + noise(5f), -65f + noise(5f), -70f + noise(5f),
                    -70f + noise(5f), -75f + noise(5f), -80f + noise(5f),
                    noise(0.2f), noise(0.2f),
                ),
            )

            ActivityContext.PHONE_CALL -> SensorSample(
                motion = floatArrayOf(
                    noise(0.1f), noise(0.1f), 9.81f + noise(0.05f),
                    noise(0.05f), noise(0.05f), noise(0.03f),
                    // Phone tilted against ear
                    3f + noise(0.5f), 8f + noise(0.5f), 4f + noise(0.5f),
                ),
                spatial = floatArrayOf(
                    30f + noise(1f), -15f + noise(1f), -48f + noise(1f),
                    37.7749f, -122.4194f, 10f,
                    1013.3f + noise(0.05f),
                ),
                visual = generateVisualFeatures(brightness = 0.1f, motion = 0.01f, edges = 0.05f), // Screen off/dark
                audio = generateAudioFeatures(type = "speech_single", energy = 0.6f),
                ambient = floatArrayOf(
                    0.05f + noise(0.02f),  // Light: blocked by face
                    0.0f,                   // Proximity: NEAR (against face!)
                    30f + noise(1f),       // Temperature: body heat
                    50f + noise(3f),
                ),
                radio = floatArrayOf(
                    -45f + noise(3f), -50f + noise(3f), -55f + noise(3f),
                    -60f + noise(3f), -65f + noise(3f), -70f + noise(3f),
                    noise(0.2f), noise(0.2f),
                ),
            )

            ActivityContext.OUTDOORS_NATURE -> SensorSample(
                motion = floatArrayOf(
                    sin(t * 3) * 1f + noise(0.3f),    // Walking on uneven terrain
                    cos(t * 3) * 0.8f + noise(0.3f),
                    9.81f + sin(t * 6) * 0.5f + noise(0.3f),
                    noise(0.1f), noise(0.1f), noise(0.1f),
                    noise(0.2f), noise(0.2f), 9.81f,
                ),
                spatial = floatArrayOf(
                    22f + noise(1f), -12f + noise(1f), -42f + noise(1f),  // Clean mag field
                    37.78f + timePhase * 0.003f, -122.42f + timePhase * 0.003f, 50f + noise(5f),
                    1012f + noise(0.5f),  // Varying altitude
                ),
                visual = generateVisualFeatures(brightness = 0.8f, motion = 0.3f, edges = 0.35f),
                audio = generateAudioFeatures(type = "nature", energy = 0.2f),
                ambient = floatArrayOf(
                    0.85f + noise(0.1f),   // Bright sunlight
                    1.0f,
                    25f + noise(3f),
                    55f + noise(10f),
                ),
                radio = floatArrayOf(
                    -90f + noise(3f), -92f + noise(3f), -95f + noise(3f),  // WiFi: very weak
                    -85f + noise(5f), -88f + noise(5f), -90f + noise(5f),  // BT: weak
                    noise(0.1f), noise(0.1f),
                ),
            )

            ActivityContext.PUBLIC_TRANSIT -> SensorSample(
                motion = floatArrayOf(
                    noise(0.4f) + sin(t * 1f) * 0.3f,  // Swaying
                    noise(0.3f) + cos(t * 0.5f) * 0.5f, // Accel/decel
                    9.81f + noise(0.2f),
                    noise(0.08f), sin(t * 0.5f) * 0.1f, noise(0.05f),
                    sin(t * 0.3f) * 0.2f, 0f, 9.81f,
                ),
                spatial = floatArrayOf(
                    35f + noise(4f), -15f + noise(4f), -50f + noise(4f), // Mag: metal vehicle
                    37.7749f + timePhase * 0.005f, -122.4194f + timePhase * 0.004f, 10f,
                    1013.2f + noise(0.1f),
                ),
                visual = generateVisualFeatures(brightness = 0.3f, motion = 0.15f, edges = 0.35f),
                audio = generateAudioFeatures(type = "crowd_transit", energy = 0.4f),
                ambient = floatArrayOf(
                    0.25f + noise(0.05f),
                    0.5f + noise(0.2f),
                    24f + noise(1f),
                    45f + noise(3f),
                ),
                radio = floatArrayOf(
                    -55f + noise(5f), -60f + noise(5f), -65f + noise(5f),
                    -50f + noise(5f), -55f + noise(5f), -60f + noise(5f), // Many BT devices
                    noise(0.3f), noise(0.3f),
                ),
            )

            ActivityContext.SHOPPING -> SensorSample(
                motion = floatArrayOf(
                    sin(t * 3) * 1.5f + noise(0.3f),   // Walking
                    cos(t * 3) * 1f + noise(0.2f),
                    9.81f + sin(t * 6) * 0.7f + noise(0.2f),
                    noise(0.1f), noise(0.1f), noise(0.15f),  // Turning to look at things
                    noise(0.1f), noise(0.1f), 9.81f,
                ),
                spatial = floatArrayOf(
                    38f + noise(2f), -10f + noise(2f), -52f + noise(2f),  // Indoor, varied
                    37.7749f + timePhase * 0.0005f, -122.4194f + timePhase * 0.0005f, 12f,
                    1013.5f + noise(0.05f),
                ),
                visual = generateVisualFeatures(brightness = 0.5f, motion = 0.3f, edges = 0.6f),
                audio = generateAudioFeatures(type = "crowd_indoor", energy = 0.35f),
                ambient = floatArrayOf(
                    0.55f + noise(0.05f),   // Bright store lighting
                    1.0f,
                    22f + noise(1f),
                    40f + noise(3f),
                ),
                radio = floatArrayOf(
                    -40f + noise(3f), -45f + noise(3f), -48f + noise(3f),  // Strong store WiFi
                    -45f + noise(3f), -48f + noise(3f), -50f + noise(3f),  // Many BT beacons
                    0.6f + noise(0.2f), 0.4f + noise(0.2f),                // UWB: store tracking
                ),
            )

            ActivityContext.CLIMBING_STAIRS -> SensorSample(
                motion = floatArrayOf(
                    sin(t * 5) * 1.5f + noise(0.4f),     // Step pattern
                    cos(t * 5) * 1f + noise(0.3f),
                    9.81f + sin(t * 10) * 2f + noise(0.3f), // Strong vertical bounce
                    sin(t * 5) * 0.3f + noise(0.1f),
                    noise(0.1f),
                    noise(0.1f),
                    noise(0.15f), noise(0.15f), 9.81f,
                ),
                spatial = floatArrayOf(
                    33f + noise(2f), -14f + noise(2f), -49f + noise(2f),
                    37.7749f, -122.4194f, 15f + timePhase * 10f,  // Altitude changing
                    1013.25f - timePhase * 1.5f,  // Barometer dropping (going up)
                ),
                visual = generateVisualFeatures(brightness = 0.3f, motion = 0.35f, edges = 0.5f),
                audio = generateAudioFeatures(type = "footsteps_indoor", energy = 0.25f),
                ambient = floatArrayOf(
                    0.25f + noise(0.05f),
                    1.0f,
                    23f + noise(1f),
                    42f + noise(3f),
                ),
                radio = floatArrayOf(
                    -50f + noise(5f), -55f + noise(5f), -60f + noise(5f),
                    -65f + noise(3f), -70f + noise(3f), -75f + noise(3f),
                    noise(0.2f), noise(0.2f),
                ),
            )

            ActivityContext.CYCLING -> SensorSample(
                motion = floatArrayOf(
                    sin(t * 4) * 0.8f + noise(0.5f),    // Pedaling rhythm
                    cos(t * 4) * 0.5f + noise(0.3f),
                    9.81f + noise(0.4f),                  // Road vibration
                    sin(t * 4) * 0.2f + noise(0.1f),    // Lean into turns
                    noise(0.15f),
                    noise(0.1f),
                    sin(t * 0.3f) * 0.5f, 0f, 9.81f,
                ),
                spatial = floatArrayOf(
                    24f + noise(2f), -11f + noise(2f), -43f + noise(2f),
                    37.7749f + timePhase * 0.008f, -122.4194f + timePhase * 0.006f, 10f + noise(2f),
                    1013.1f + noise(0.3f),
                ),
                visual = generateVisualFeatures(brightness = 0.7f, motion = 0.55f, edges = 0.4f),
                audio = generateAudioFeatures(type = "wind", energy = 0.35f),
                ambient = floatArrayOf(
                    0.75f + noise(0.1f),   // Outdoors
                    1.0f,
                    24f + noise(3f),
                    50f + noise(8f),
                ),
                radio = floatArrayOf(
                    -80f + noise(8f), -85f + noise(8f), -88f + noise(8f),
                    -50f + noise(3f), -55f + noise(3f), -60f + noise(3f),  // BT earbuds
                    noise(0.2f), noise(0.2f),
                ),
            )

            ActivityContext.IDLE_TABLE -> SensorSample(
                motion = floatArrayOf(
                    noise(0.01f), noise(0.01f), 9.81f + noise(0.005f),
                    noise(0.002f), noise(0.002f), noise(0.002f),
                    0f, 0f, 9.81f,
                ),
                spatial = floatArrayOf(
                    30f + noise(0.3f), -14f + noise(0.3f), -48f + noise(0.3f),
                    37.7749f, -122.4194f, 10f,
                    1013.4f + noise(0.02f),
                ),
                visual = generateVisualFeatures(brightness = 0.35f, motion = 0.01f, edges = 0.2f),
                audio = generateAudioFeatures(type = "ambient_quiet", energy = 0.05f),
                ambient = floatArrayOf(
                    0.3f + noise(0.02f),
                    0.25f + noise(0.05f),  // Face down on table
                    22f + noise(0.5f),
                    42f + noise(2f),
                ),
                radio = floatArrayOf(
                    -42f + noise(1f), -47f + noise(1f), -52f + noise(1f),
                    -62f + noise(2f), -67f + noise(2f), -72f + noise(2f),
                    noise(0.1f), noise(0.1f),
                ),
            )

            ActivityContext.TRANSITION -> {
                // Shouldn't be called directly, but provide default
                generateSingleSample(ActivityContext.WALKING, timePhase)
            }
        }
    }

    /**
     * Generate a 64-dim visual feature vector matching FrontCameraProcessor output format.
     */
    private fun generateVisualFeatures(
        brightness: Float,
        motion: Float,
        edges: Float,
    ): FloatArray {
        val noise = { scale: Float -> (rng.nextFloat() - 0.5f) * 2 * scale }
        return FloatArray(64) { i ->
            when {
                i < 8 -> brightness + noise(0.1f)       // Brightness distribution
                i < 20 -> {                               // Color histogram
                    val bin = i - 8
                    val center = brightness * 12
                    val dist = abs(bin - center)
                    (1f / (1f + dist)) + noise(0.05f)
                }
                i < 28 -> motion + noise(0.05f)          // Motion estimation
                i < 44 -> edges + noise(0.1f)             // Edge density
                i < 52 -> {                                // Face/person presence
                    if (brightness > 0.1f) 0.5f + noise(0.15f) else noise(0.02f)
                }
                else -> edges * 0.8f + noise(0.08f)       // Texture complexity
            }.coerceIn(0f, 1f)
        }
    }

    /**
     * Generate 40-dim MFCC-like audio features for different sound environments.
     */
    private fun generateAudioFeatures(type: String, energy: Float): FloatArray {
        val noise = { scale: Float -> (rng.nextFloat() - 0.5f) * 2 * scale }
        return FloatArray(40) { i ->
            val base = when (type) {
                "footsteps_outdoor" -> if (i < 5) energy * 1.5f else energy * 0.3f * exp(-i * 0.1f)
                "office" -> energy * 0.5f * exp(-i * 0.05f) + if (i in 3..8) 0.1f else 0f
                "driving" -> energy * (if (i < 3) 1.2f else 0.6f) * exp(-i * 0.03f)
                "speech_single" -> if (i in 2..15) energy * 0.8f else energy * 0.1f
                "speech_multiple" -> if (i in 2..18) energy * 0.9f else energy * 0.15f
                "silence" -> energy * 0.1f * exp(-i * 0.2f)
                "kitchen" -> energy * 0.7f * exp(-i * 0.04f) + if (i in 5..12) 0.15f else 0f
                "exercise" -> energy * (if (i < 8) 1f else 0.5f) * exp(-i * 0.05f)
                "elevator" -> energy * 0.3f * exp(-i * 0.08f) + if (i < 3) 0.2f else 0f
                "nature" -> energy * 0.6f * exp(-i * 0.06f) + sin(i * 0.5f) * 0.05f
                "crowd_transit" -> energy * 0.8f * exp(-i * 0.04f)
                "crowd_indoor" -> energy * 0.7f * exp(-i * 0.045f)
                "footsteps_indoor" -> if (i < 6) energy * 1.2f else energy * 0.25f * exp(-i * 0.08f)
                "wind" -> if (i < 4) energy * 1.5f else energy * 0.2f * exp(-i * 0.1f)
                "ambient_quiet" -> energy * 0.2f * exp(-i * 0.15f)
                else -> energy * 0.5f * exp(-i * 0.05f)
            }
            (base + noise(energy * 0.1f)).coerceIn(-5f, 5f)
        }
    }

    /**
     * Blend two sensor samples together (for transition generation).
     */
    private fun blendSamples(a: SensorSample, b: SensorSample, blend: Float): SensorSample {
        fun blendArrays(x: FloatArray, y: FloatArray): FloatArray {
            return FloatArray(x.size) { i -> x[i] * (1 - blend) + y[i] * blend }
        }
        return SensorSample(
            motion = blendArrays(a.motion, b.motion),
            spatial = blendArrays(a.spatial, b.spatial),
            visual = blendArrays(a.visual, b.visual),
            audio = blendArrays(a.audio, b.audio),
            ambient = blendArrays(a.ambient, b.ambient),
            radio = blendArrays(a.radio, b.radio),
        )
    }

    /**
     * Generate an anomalous sequence — normal context but with random sensor spikes/drops.
     */
    private fun generateAnomalousSequence(
        context: ActivityContext,
        length: Int,
    ): List<SensorSample> {
        return (0 until length).map { t ->
            val sample = generateSingleSample(context, t.toFloat() / length)
            // Inject random anomalies
            val anomalyType = rng.nextInt(4)
            when (anomalyType) {
                0 -> sample.copy(motion = sample.motion.map { it * (2f + rng.nextFloat() * 3f) }.toFloatArray())
                1 -> sample.copy(audio = sample.audio.map { it * (3f + rng.nextFloat() * 2f) }.toFloatArray())
                2 -> sample.copy(ambient = sample.ambient.map { it * rng.nextFloat() * 0.1f }.toFloatArray())
                3 -> sample.copy(spatial = sample.spatial.mapIndexed { i, v ->
                    if (i == 6) v + (rng.nextFloat() - 0.5f) * 20f else v  // Pressure spike
                }.toFloatArray())
                else -> sample
            }
        }
    }
}
