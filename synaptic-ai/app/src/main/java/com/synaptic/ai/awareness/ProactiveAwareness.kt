package com.synaptic.ai.awareness

import com.synaptic.ai.core.SensorSynesthesia
import com.synaptic.ai.memory.PersistentMemory
import com.synaptic.ai.model.SynapticTransformer

/**
 * ProactiveAwareness: The "consciousness" layer of SynapticAI.
 *
 * This is what makes SynapticAI fundamentally different from every existing AI:
 * IT DOESN'T WAIT TO BE ASKED.
 *
 * Current AI (ChatGPT, Siri, Alexa) is purely reactive — it sits dormant until
 * you speak to it. SynapticAI continuously builds an understanding of your world
 * and proactively surfaces insights when they're relevant.
 *
 * The awareness engine combines:
 * 1. Model output (what the transformer thinks is happening)
 * 2. Synesthetic perceptions (cross-sensor emergent understanding)
 * 3. Memory context (what usually happens, what's different now)
 *
 * And produces AWARENESS EVENTS — things the AI notices that might matter to you:
 *
 * - "You usually leave for work by now" (routine deviation)
 * - "The barometric pressure is dropping fast — weather change coming" (sensor insight)
 * - "This place feels like the coffee shop but you've never been here" (place recognition)
 * - "Your movement pattern suggests you're stressed" (behavioral inference)
 * - "Someone nearby is talking to you" (social awareness)
 * - "You've been sitting for 2 hours" (health awareness)
 * - "You're in a new building — mapping the space" (exploration)
 *
 * Events are ranked by relevance and novelty, and only surfaced when they cross
 * a user-configurable attention threshold.
 */
class ProactiveAwareness {

    private var lastEventTime = 0L
    private val recentEvents = mutableListOf<AwarenessEvent>()
    private val eventCooldowns = mutableMapOf<EventType, Long>()
    private var attentionThreshold = 0.5f  // User-configurable

    // State tracking for duration-based events
    private var lastMotionTime = System.currentTimeMillis()
    private var lastStandingTime = System.currentTimeMillis()
    private var sittingStartTime: Long? = null
    private var currentActivityConfidence = 0f

    enum class EventType {
        ROUTINE_DEVIATION,       // You're off your normal schedule
        WEATHER_CHANGE,          // Barometric pressure shift detected
        PLACE_RECOGNITION,       // Recognized or new place
        BEHAVIORAL_PATTERN,      // Detected activity pattern change
        SOCIAL_AWARENESS,        // Social environment change
        HEALTH_REMINDER,         // Sedentary too long, etc.
        SPATIAL_EXPLORATION,     // In a new/unknown space
        TRANSITION_DETECTED,     // Moving between contexts
        ANOMALY_ALERT,           // Something unusual happening
        CONTEXT_PREDICTION,      // Predicting what's next
    }

    enum class EventPriority { LOW, MEDIUM, HIGH, CRITICAL }

    data class AwarenessEvent(
        val type: EventType,
        val priority: EventPriority,
        val title: String,
        val description: String,
        val confidence: Float,
        val timestamp: Long,
        val relatedSensorData: Map<String, Float>? = null,
    )

    /**
     * Main awareness loop: analyze current state and generate events.
     * Called at ~1Hz (downsampled from the 10Hz inference loop).
     */
    fun analyze(
        modelOutput: SynapticTransformer.ModelOutput,
        perception: SensorSynesthesia.SynestheticPerception,
        memoryContext: PersistentMemory.MemoryContext,
    ): List<AwarenessEvent> {
        val now = System.currentTimeMillis()
        val events = mutableListOf<AwarenessEvent>()

        // --- Check each awareness domain ---

        checkRoutineDeviation(memoryContext, now)?.let { events.add(it) }
        checkWeatherChange(perception, now)?.let { events.add(it) }
        checkPlaceAwareness(memoryContext, perception, now)?.let { events.add(it) }
        checkBehavioralPattern(modelOutput, perception, now)?.let { events.add(it) }
        checkSocialAwareness(perception, now)?.let { events.add(it) }
        checkHealthReminders(perception, now)?.let { events.add(it) }
        checkTransitions(perception, memoryContext, now)?.let { events.add(it) }
        checkAnomalies(modelOutput, perception, now)?.let { events.add(it) }

        // Filter by attention threshold and cooldowns
        val filteredEvents = events.filter { event ->
            event.confidence >= attentionThreshold &&
                now - (eventCooldowns[event.type] ?: 0L) > getCooldownMs(event.type)
        }

        // Update cooldowns for emitted events
        for (event in filteredEvents) {
            eventCooldowns[event.type] = now
        }

        // Keep recent events bounded
        recentEvents.addAll(filteredEvents)
        while (recentEvents.size > 100) recentEvents.removeAt(0)

        return filteredEvents
    }

    private fun checkRoutineDeviation(
        memory: PersistentMemory.MemoryContext,
        now: Long,
    ): AwarenessEvent? {
        if (memory.routineDeviation < 0.5f) return null
        val expected = memory.expectedActivity ?: return null

        return AwarenessEvent(
            type = EventType.ROUTINE_DEVIATION,
            priority = EventPriority.LOW,
            title = "Off routine",
            description = "You're usually at place #${expected.typicalPlaceId} at this time",
            confidence = memory.routineDeviation * expected.confidence,
            timestamp = now,
        )
    }

    private fun checkWeatherChange(
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        // Environmental mood index 0 is barometric pressure deviation
        val pressureChange = perception.environmentalMood.getOrElse(0) { 0f }
        if (kotlin.math.abs(pressureChange) < 0.3f) return null

        val direction = if (pressureChange < 0) "dropping" else "rising"
        return AwarenessEvent(
            type = EventType.WEATHER_CHANGE,
            priority = EventPriority.LOW,
            title = "Pressure $direction",
            description = "Barometric pressure is $direction — possible weather change",
            confidence = kotlin.math.abs(pressureChange).coerceIn(0f, 1f),
            timestamp = now,
            relatedSensorData = mapOf("pressure_delta" to pressureChange),
        )
    }

    private fun checkPlaceAwareness(
        memory: PersistentMemory.MemoryContext,
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        if (memory.isKnownLocation) {
            val place = memory.currentPlace ?: return null
            if (place.visitCount <= 2) {
                return AwarenessEvent(
                    type = EventType.PLACE_RECOGNITION,
                    priority = EventPriority.LOW,
                    title = "Recently visited place",
                    description = "You've been here ${place.visitCount} times before (${place.name ?: "unnamed"})",
                    confidence = 0.7f,
                    timestamp = now,
                )
            }
            return null
        }

        // Unknown location with low novelty = possibly a new routine place
        if (perception.overallNovelty < 0.3f) {
            return AwarenessEvent(
                type = EventType.SPATIAL_EXPLORATION,
                priority = EventPriority.MEDIUM,
                title = "New place detected",
                description = "You're in an unfamiliar space. Learning its sensor signature.",
                confidence = 0.8f,
                timestamp = now,
            )
        }
        return null
    }

    private fun checkBehavioralPattern(
        modelOutput: SynapticTransformer.ModelOutput,
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        // Detect significant rhythm changes
        val rhythmChange = perception.temporalRhythm.map { kotlin.math.abs(it) }.average().toFloat()
        if (rhythmChange < 0.4f) return null

        return AwarenessEvent(
            type = EventType.BEHAVIORAL_PATTERN,
            priority = EventPriority.LOW,
            title = "Activity shift",
            description = "Your activity rhythm has changed significantly",
            confidence = rhythmChange.coerceIn(0f, 1f),
            timestamp = now,
        )
    }

    private fun checkSocialAwareness(
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        val socialDensity = perception.socialDensity
        val speechLevel = socialDensity.getOrElse(0) { 0f }
        val radioLevel = socialDensity.getOrElse(1) { 0f }

        // Someone is speaking nearby
        if (speechLevel > 0.6f) {
            return AwarenessEvent(
                type = EventType.SOCIAL_AWARENESS,
                priority = EventPriority.MEDIUM,
                title = "Speech detected",
                description = "Active conversation detected nearby",
                confidence = speechLevel,
                timestamp = now,
                relatedSensorData = mapOf(
                    "speech_level" to speechLevel,
                    "device_density" to radioLevel,
                ),
            )
        }
        return null
    }

    private fun checkHealthReminders(
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        val motionEnergy = perception.temporalRhythm.getOrElse(0) { 0f }

        // Track sedentary time
        if (motionEnergy > 0.1f) {
            lastMotionTime = now
            sittingStartTime = null
        } else {
            if (sittingStartTime == null) sittingStartTime = now
        }

        val sittingDuration = sittingStartTime?.let { now - it } ?: 0L
        val twoHoursMs = 2 * 60 * 60 * 1000L

        if (sittingDuration > twoHoursMs) {
            return AwarenessEvent(
                type = EventType.HEALTH_REMINDER,
                priority = EventPriority.MEDIUM,
                title = "Time to move",
                description = "You've been sedentary for ${sittingDuration / 60000}+ minutes",
                confidence = 0.9f,
                timestamp = now,
                relatedSensorData = mapOf(
                    "sitting_minutes" to (sittingDuration / 60000f),
                    "motion_energy" to motionEnergy,
                ),
            )
        }
        return null
    }

    private fun checkTransitions(
        perception: SensorSynesthesia.SynestheticPerception,
        memory: PersistentMemory.MemoryContext,
        now: Long,
    ): AwarenessEvent? {
        if (!perception.isTransitioning) return null

        return AwarenessEvent(
            type = EventType.TRANSITION_DETECTED,
            priority = EventPriority.LOW,
            title = "Context change",
            description = "Transitioning between environments",
            confidence = perception.transitionConfidence,
            timestamp = now,
        )
    }

    private fun checkAnomalies(
        modelOutput: SynapticTransformer.ModelOutput,
        perception: SensorSynesthesia.SynestheticPerception,
        now: Long,
    ): AwarenessEvent? {
        if (modelOutput.anomalyScore < 0.6f) return null

        return AwarenessEvent(
            type = EventType.ANOMALY_ALERT,
            priority = if (modelOutput.anomalyScore > 0.8f) EventPriority.HIGH else EventPriority.MEDIUM,
            title = "Unusual activity",
            description = "Sensor patterns indicate something unusual in your environment",
            confidence = modelOutput.anomalyScore,
            timestamp = now,
        )
    }

    private fun getCooldownMs(type: EventType): Long = when (type) {
        EventType.ROUTINE_DEVIATION -> 30 * 60 * 1000L    // 30 min
        EventType.WEATHER_CHANGE -> 60 * 60 * 1000L       // 1 hour
        EventType.PLACE_RECOGNITION -> 5 * 60 * 1000L     // 5 min
        EventType.BEHAVIORAL_PATTERN -> 10 * 60 * 1000L   // 10 min
        EventType.SOCIAL_AWARENESS -> 2 * 60 * 1000L      // 2 min
        EventType.HEALTH_REMINDER -> 30 * 60 * 1000L      // 30 min
        EventType.SPATIAL_EXPLORATION -> 5 * 60 * 1000L    // 5 min
        EventType.TRANSITION_DETECTED -> 60 * 1000L        // 1 min
        EventType.ANOMALY_ALERT -> 5 * 60 * 1000L         // 5 min
        EventType.CONTEXT_PREDICTION -> 15 * 60 * 1000L   // 15 min
    }

    fun setAttentionThreshold(threshold: Float) {
        attentionThreshold = threshold.coerceIn(0.1f, 0.95f)
    }

    fun getRecentEvents(count: Int = 10): List<AwarenessEvent> {
        return recentEvents.takeLast(count)
    }
}
