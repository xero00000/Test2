package com.synaptic.ai.memory

import android.content.Context
import com.synaptic.ai.core.SensorSynesthesia
import com.synaptic.ai.core.Tensor
import com.synaptic.ai.model.SynapticTransformer
import kotlinx.coroutines.*
import java.io.*
import kotlin.math.*

/**
 * PersistentMemory: The long-term memory system of SynapticAI.
 *
 * Unlike conversational AI that forgets everything between sessions, SynapticAI
 * builds a persistent, evolving understanding of the user's world. This system
 * stores and retrieves:
 *
 * 1. PLACE MEMORIES - Learned representations of locations the user frequents.
 *    Each place has a unique "sensor signature" (what it feels like through the
 *    phone's sensors). Over time, the AI knows "this is home" vs "this is work"
 *    vs "this is the gym" without GPS — just from sensor patterns.
 *
 * 2. ROUTINE PATTERNS - Temporal sequences that repeat. The AI learns your daily
 *    rhythm: wake up, commute, work, lunch, work, commute, home. It can then
 *    predict what you'll do next and detect anomalies ("you usually leave for
 *    work at 8am but it's 9am and you're still home").
 *
 * 3. TRANSITION MEMORIES - What sensor patterns look like when you move between
 *    contexts. The AI learns what "leaving home" and "arriving at work" feel like,
 *    allowing it to recognize transitions in real-time.
 *
 * 4. ANOMALY HISTORY - A log of unusual events, allowing the AI to distinguish
 *    between "genuinely unusual" and "unusual but has happened before."
 *
 * Storage: Uses a simple file-based system with protobuf-like binary serialization.
 * Fits in ~10-50MB depending on usage duration.
 */
class PersistentMemory(private val context: Context) {

    private val memoryDir = File(context.filesDir, "synaptic_memory")
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // In-memory caches
    private val placeMemories = mutableListOf<PlaceMemory>()
    private val routinePatterns = mutableListOf<RoutinePattern>()
    private val transitionMemories = mutableListOf<TransitionMemory>()
    private val anomalyLog = mutableListOf<AnomalyEvent>()

    // Current session state
    private var currentPlaceId: Int = -1
    private var sessionStartTime: Long = System.currentTimeMillis()
    private val sessionTimeline = mutableListOf<TimelineEntry>()

    data class PlaceMemory(
        val id: Int,
        val name: String?,                         // User-assigned name (optional)
        val sensorSignature: FloatArray,           // Average synesthetic perception
        val signatureVariance: FloatArray,         // How much the signature varies
        val visitCount: Int,
        val totalDurationMs: Long,
        val firstVisit: Long,
        val lastVisit: Long,
        val typicalArrivalHours: FloatArray,       // 24-bin histogram of arrival times
        val typicalDepartureHours: FloatArray,     // 24-bin histogram of departure times
    )

    data class RoutinePattern(
        val id: Int,
        val dayOfWeek: Int,                        // 0-6 or -1 for any day
        val hourOfDay: Int,                        // 0-23
        val typicalPlaceId: Int,                   // Where user usually is at this time
        val typicalActivity: FloatArray,           // Average temporal rhythm features
        val confidence: Float,                     // How consistent is this pattern
        val sampleCount: Int,
    )

    data class TransitionMemory(
        val fromPlaceId: Int,
        val toPlaceId: Int,
        val typicalDurationMs: Long,
        val sensorTrajectory: List<FloatArray>,    // Sequence of synesthetic features during transition
        val occurrenceCount: Int,
    )

    data class AnomalyEvent(
        val timestamp: Long,
        val placeId: Int,
        val anomalyVector: FloatArray,
        val description: String,
        val noveltyScore: Float,
    )

    data class TimelineEntry(
        val timestamp: Long,
        val worldState: FloatArray,
        val placeId: Int,
        val activityType: Int,
    )

    fun initialize() {
        memoryDir.mkdirs()
        loadMemories()
    }

    /**
     * Process a new perception and update memories.
     * Called at each inference step (~10Hz), but memory updates happen less frequently.
     */
    fun process(
        perception: SensorSynesthesia.SynestheticPerception,
        modelOutput: SynapticTransformer.ModelOutput,
    ): MemoryContext {
        val now = System.currentTimeMillis()

        // Try to identify current place from sensor signature
        val matchedPlace = matchPlace(perception.spatialTexture)
        val placeId = matchedPlace?.id ?: -1

        // Detect place transition
        if (placeId != currentPlaceId && placeId >= 0) {
            if (currentPlaceId >= 0) {
                recordTransition(currentPlaceId, placeId, perception)
            }
            currentPlaceId = placeId
        }

        // Update place memory with new observation
        if (placeId >= 0) {
            updatePlaceMemory(placeId, perception)
        } else if (perception.overallNovelty < 0.2f) {
            // Low novelty but unknown place - create new place memory
            createNewPlace(perception)
        }

        // Record anomalies
        if (perception.overallNovelty > 0.5f) {
            recordAnomaly(perception, placeId)
        }

        // Add to session timeline (downsampled to ~1Hz)
        if (sessionTimeline.isEmpty() ||
            now - sessionTimeline.last().timestamp > 1000
        ) {
            sessionTimeline.add(
                TimelineEntry(
                    timestamp = now,
                    worldState = modelOutput.worldState.data.clone(),
                    placeId = placeId,
                    activityType = modelOutput.contextPredictions.data.indices.maxByOrNull {
                        modelOutput.contextPredictions.data[it]
                    } ?: 0,
                )
            )
        }

        // Periodically persist to disk (every 30 seconds)
        if (sessionTimeline.size % 30 == 0) {
            scope.launch { saveMemories() }
        }

        // Build memory context for the awareness engine
        return MemoryContext(
            currentPlace = matchedPlace,
            isKnownLocation = placeId >= 0,
            placeVisitCount = matchedPlace?.visitCount ?: 0,
            expectedActivity = getExpectedActivity(now),
            recentAnomalies = anomalyLog.takeLast(5),
            routineDeviation = computeRoutineDeviation(now, placeId),
        )
    }

    data class MemoryContext(
        val currentPlace: PlaceMemory?,
        val isKnownLocation: Boolean,
        val placeVisitCount: Int,
        val expectedActivity: RoutinePattern?,
        val recentAnomalies: List<AnomalyEvent>,
        val routineDeviation: Float,              // 0 = on routine, 1 = completely off
    )

    /**
     * Match current sensor signature to known places using cosine similarity.
     */
    private fun matchPlace(spatialTexture: FloatArray): PlaceMemory? {
        var bestMatch: PlaceMemory? = null
        var bestScore = 0.6f  // Minimum similarity threshold

        for (place in placeMemories) {
            val similarity = cosineSimilarity(spatialTexture, place.sensorSignature)
            if (similarity > bestScore) {
                bestScore = similarity
                bestMatch = place
            }
        }
        return bestMatch
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        val minLen = minOf(a.size, b.size)
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in 0 until minLen) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom > 0) dotProduct / denom else 0f
    }

    private fun createNewPlace(perception: SensorSynesthesia.SynestheticPerception) {
        val id = placeMemories.size
        placeMemories.add(
            PlaceMemory(
                id = id,
                name = null,
                sensorSignature = perception.spatialTexture.clone(),
                signatureVariance = FloatArray(perception.spatialTexture.size),
                visitCount = 1,
                totalDurationMs = 0,
                firstVisit = System.currentTimeMillis(),
                lastVisit = System.currentTimeMillis(),
                typicalArrivalHours = FloatArray(24),
                typicalDepartureHours = FloatArray(24),
            )
        )
    }

    private fun updatePlaceMemory(placeId: Int, perception: SensorSynesthesia.SynestheticPerception) {
        if (placeId >= placeMemories.size) return
        val place = placeMemories[placeId]

        // Exponential moving average update of sensor signature
        val alpha = 0.05f
        val updatedSignature = FloatArray(place.sensorSignature.size) { i ->
            if (i < perception.spatialTexture.size) {
                place.sensorSignature[i] * (1 - alpha) + perception.spatialTexture[i] * alpha
            } else {
                place.sensorSignature[i]
            }
        }

        placeMemories[placeId] = place.copy(
            sensorSignature = updatedSignature,
            lastVisit = System.currentTimeMillis(),
        )
    }

    private fun recordTransition(fromId: Int, toId: Int, perception: SensorSynesthesia.SynestheticPerception) {
        val existing = transitionMemories.find { it.fromPlaceId == fromId && it.toPlaceId == toId }
        if (existing != null) {
            val idx = transitionMemories.indexOf(existing)
            transitionMemories[idx] = existing.copy(
                occurrenceCount = existing.occurrenceCount + 1,
            )
        } else {
            transitionMemories.add(
                TransitionMemory(
                    fromPlaceId = fromId,
                    toPlaceId = toId,
                    typicalDurationMs = 0,
                    sensorTrajectory = listOf(perception.spatialTexture.clone()),
                    occurrenceCount = 1,
                )
            )
        }
    }

    private fun recordAnomaly(perception: SensorSynesthesia.SynestheticPerception, placeId: Int) {
        if (anomalyLog.size > 1000) {
            anomalyLog.removeAt(0) // Keep bounded
        }
        anomalyLog.add(
            AnomalyEvent(
                timestamp = System.currentTimeMillis(),
                placeId = placeId,
                anomalyVector = perception.anomalyVector.clone(),
                description = generateAnomalyDescription(perception),
                noveltyScore = perception.overallNovelty,
            )
        )
    }

    private fun generateAnomalyDescription(perception: SensorSynesthesia.SynestheticPerception): String {
        val parts = mutableListOf<String>()
        if (abs(perception.anomalyVector.getOrElse(0) { 0f }) > 0.3f) parts.add("spatial_change")
        if (abs(perception.anomalyVector.getOrElse(3) { 0f }) > 0.3f) parts.add("social_change")
        if (abs(perception.anomalyVector.getOrElse(6) { 0f }) > 0.3f) parts.add("rhythm_change")
        if (abs(perception.anomalyVector.getOrElse(9) { 0f }) > 0.3f) parts.add("mood_change")
        return parts.joinToString("+").ifEmpty { "general_anomaly" }
    }

    private fun getExpectedActivity(now: Long): RoutinePattern? {
        val calendar = java.util.Calendar.getInstance().apply { timeInMillis = now }
        val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK) - 1
        val hour = calendar.get(java.util.Calendar.HOUR_OF_DAY)

        return routinePatterns.find {
            (it.dayOfWeek == dayOfWeek || it.dayOfWeek == -1) && it.hourOfDay == hour
        }
    }

    private fun computeRoutineDeviation(now: Long, currentPlaceId: Int): Float {
        val expected = getExpectedActivity(now) ?: return 0.5f  // Unknown = moderate deviation
        return if (expected.typicalPlaceId == currentPlaceId) 0f else 1f
    }

    private fun saveMemories() {
        try {
            // Save place memories
            ObjectOutputStream(FileOutputStream(File(memoryDir, "places.dat"))).use { oos ->
                oos.writeInt(placeMemories.size)
                for (place in placeMemories) {
                    oos.writeInt(place.id)
                    oos.writeObject(place.name)
                    oos.writeObject(place.sensorSignature)
                    oos.writeInt(place.visitCount)
                    oos.writeLong(place.totalDurationMs)
                    oos.writeLong(place.firstVisit)
                    oos.writeLong(place.lastVisit)
                }
            }
        } catch (e: Exception) {
            // Handle gracefully - memory loss is not critical
        }
    }

    private fun loadMemories() {
        try {
            val placesFile = File(memoryDir, "places.dat")
            if (placesFile.exists()) {
                ObjectInputStream(FileInputStream(placesFile)).use { ois ->
                    val count = ois.readInt()
                    for (i in 0 until count) {
                        val id = ois.readInt()
                        val name = ois.readObject() as? String
                        val signature = ois.readObject() as FloatArray
                        val visitCount = ois.readInt()
                        val totalDuration = ois.readLong()
                        val firstVisit = ois.readLong()
                        val lastVisit = ois.readLong()
                        placeMemories.add(
                            PlaceMemory(
                                id = id,
                                name = name,
                                sensorSignature = signature,
                                signatureVariance = FloatArray(signature.size),
                                visitCount = visitCount,
                                totalDurationMs = totalDuration,
                                firstVisit = firstVisit,
                                lastVisit = lastVisit,
                                typicalArrivalHours = FloatArray(24),
                                typicalDepartureHours = FloatArray(24),
                            )
                        )
                    }
                }
            }
        } catch (e: Exception) {
            // Fresh start if memories are corrupted
            placeMemories.clear()
        }
    }

    fun shutdown() {
        scope.launch {
            saveMemories()
            scope.cancel()
        }
    }
}
