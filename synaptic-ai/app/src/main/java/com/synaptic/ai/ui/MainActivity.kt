package com.synaptic.ai.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.synaptic.ai.SynapticApplication
import com.synaptic.ai.awareness.ProactiveAwareness
import com.synaptic.ai.runtime.InferenceRuntime
import com.synaptic.ai.sensors.FrontCameraProcessor
import com.synaptic.ai.sensors.SensorFusionService
import kotlinx.coroutines.flow.collectLatest

class MainActivity : ComponentActivity() {

    private lateinit var runtime: InferenceRuntime
    private var cameraProcessor: FrontCameraProcessor? = null

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACTIVITY_RECOGNITION,
        Manifest.permission.BODY_SENSORS,
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.values.any { it }) {
            startSynapticAI()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        runtime = (application as SynapticApplication).runtime

        setContent {
            SynapticTheme {
                SynapticMainScreen(runtime)
            }
        }

        // Request permissions
        val needed = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isEmpty()) {
            startSynapticAI()
        } else {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun startSynapticAI() {
        // Start foreground service for continuous sensing
        val serviceIntent = Intent(this, SensorFusionService::class.java)
        startForegroundService(serviceIntent)

        // Start the inference runtime
        runtime.start()

        // Start front camera processing (front camera only per user request)
        cameraProcessor = FrontCameraProcessor(this) { features ->
            runtime.getSensorEngine().updateVisualFeatures(features)
        }
        cameraProcessor?.start(this)
    }

    override fun onDestroy() {
        cameraProcessor?.stop()
        super.onDestroy()
    }
}

@Composable
fun SynapticTheme(content: @Composable () -> Unit) {
    val darkColorScheme = darkColorScheme(
        primary = Color(0xFF7DF9FF),       // Electric cyan
        secondary = Color(0xFFBF40BF),     // Magenta
        tertiary = Color(0xFF39FF14),      // Neon green
        background = Color(0xFF0A0A0F),    // Near black
        surface = Color(0xFF12121A),       // Dark surface
        onPrimary = Color.Black,
        onSecondary = Color.White,
        onBackground = Color.White,
        onSurface = Color.White,
    )

    MaterialTheme(
        colorScheme = darkColorScheme,
        content = content,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SynapticMainScreen(runtime: InferenceRuntime) {
    val state by runtime.stateFlow.collectAsState()
    val events = remember { mutableStateListOf<ProactiveAwareness.AwarenessEvent>() }

    LaunchedEffect(Unit) {
        runtime.eventFlow.collectLatest { event ->
            events.add(0, event) // Newest first
            if (events.size > 50) events.removeLast()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF0A0A0F),
                        Color(0xFF0D0D1A),
                        Color(0xFF0A0F0A),
                    )
                )
            )
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Header
            item {
                SynapticHeader(state)
            }

            // Status orb
            item {
                StatusOrb(state)
            }

            // Active sensor experts
            item {
                ActiveExpertsCard(state)
            }

            // Performance metrics
            item {
                MetricsCard(state)
            }

            // Awareness events
            item {
                Text(
                    "Awareness Feed",
                    color = Color(0xFF7DF9FF),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            if (events.isEmpty()) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF12121A)),
                    ) {
                        Text(
                            "SynapticAI is learning your environment...\nAwareness events will appear here.",
                            modifier = Modifier.padding(20.dp),
                            color = Color.Gray,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }

            items(events) { event ->
                AwarenessEventCard(event)
            }

            // Bottom spacer
            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@Composable
fun SynapticHeader(state: InferenceRuntime.RuntimeState) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 32.dp, bottom = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "SYNAPTIC",
            fontSize = 32.sp,
            fontWeight = FontWeight.Black,
            color = Color(0xFF7DF9FF),
            letterSpacing = 8.sp,
        )
        Text(
            "AI",
            fontSize = 14.sp,
            fontWeight = FontWeight.Light,
            color = Color(0xFF7DF9FF).copy(alpha = 0.6f),
            letterSpacing = 12.sp,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            if (state.isRunning) "Environmental Awareness Active" else "Initializing...",
            fontSize = 12.sp,
            color = if (state.isRunning) Color(0xFF39FF14) else Color.Gray,
        )
    }
}

@Composable
fun StatusOrb(state: InferenceRuntime.RuntimeState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12121A)),
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // Anomaly indicator
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                getAnomalyColor(state.anomalyScore),
                                getAnomalyColor(state.anomalyScore).copy(alpha = 0.1f),
                            )
                        )
                    ),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "${(state.confidence * 100).toInt()}%",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                "Place: ${state.currentPlace}",
                color = Color.White.copy(alpha = 0.8f),
                fontSize = 14.sp,
            )

            if (state.isTransitioning) {
                Text(
                    "Transitioning...",
                    color = Color(0xFFBF40BF),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                )
            }

            Text(
                "Novelty: ${(state.overallNovelty * 100).toInt()}%",
                color = Color.White.copy(alpha = 0.5f),
                fontSize = 12.sp,
            )
        }
    }
}

@Composable
fun ActiveExpertsCard(state: InferenceRuntime.RuntimeState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12121A)),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Active Sensor Experts",
                color = Color(0xFF7DF9FF),
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val allExperts = listOf("MOTION", "SPATIAL", "VISUAL", "AUDIO", "AMBIENT", "RADIO")
                for (expert in allExperts) {
                    val isActive = expert in state.activeExperts
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (isActive) Color(0xFF39FF14).copy(alpha = 0.2f)
                                else Color.White.copy(alpha = 0.05f)
                            )
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            expert.take(3),
                            fontSize = 10.sp,
                            fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal,
                            color = if (isActive) Color(0xFF39FF14) else Color.Gray,
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun MetricsCard(state: InferenceRuntime.RuntimeState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12121A)),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Performance",
                color = Color(0xFF7DF9FF),
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(modifier = Modifier.fillMaxWidth()) {
                MetricItem("Inferences", "${state.inferenceCount}", Modifier.weight(1f))
                MetricItem("Avg Time", "${String.format("%.1f", state.avgInferenceTimeMs)}ms", Modifier.weight(1f))
                MetricItem("Efficiency", state.batteryEfficiency.substringBefore(" "), Modifier.weight(1f))
            }
        }
    }
}

@Composable
fun MetricItem(label: String, value: String, modifier: Modifier) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(value, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp)
        Text(label, color = Color.Gray, fontSize = 10.sp)
    }
}

@Composable
fun AwarenessEventCard(event: ProactiveAwareness.AwarenessEvent) {
    val borderColor = when (event.priority) {
        ProactiveAwareness.EventPriority.CRITICAL -> Color(0xFFFF0040)
        ProactiveAwareness.EventPriority.HIGH -> Color(0xFFFF6600)
        ProactiveAwareness.EventPriority.MEDIUM -> Color(0xFFBF40BF)
        ProactiveAwareness.EventPriority.LOW -> Color(0xFF7DF9FF).copy(alpha = 0.5f)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12121A)),
        shape = RoundedCornerShape(12.dp),
        border = CardDefaults.outlinedCardBorder().copy(
            width = 1.dp,
            brush = Brush.linearGradient(listOf(borderColor, borderColor.copy(alpha = 0.2f))),
        ),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    event.title,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                )
                Text(
                    "${(event.confidence * 100).toInt()}%",
                    color = borderColor,
                    fontSize = 12.sp,
                )
            }
            Text(
                event.description,
                color = Color.White.copy(alpha = 0.7f),
                fontSize = 12.sp,
                modifier = Modifier.padding(top = 4.dp),
            )
            Text(
                event.type.name.lowercase().replace("_", " "),
                color = Color.Gray,
                fontSize = 10.sp,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

private fun getAnomalyColor(score: Float): Color {
    return when {
        score < 0.3f -> Color(0xFF39FF14)   // Green - normal
        score < 0.6f -> Color(0xFF7DF9FF)   // Cyan - mild
        score < 0.8f -> Color(0xFFBF40BF)   // Magenta - notable
        else -> Color(0xFFFF0040)            // Red - anomalous
    }
}
