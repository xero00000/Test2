plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.synaptic.ai"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.synaptic.ai"
        minSdk = 31  // Android 12+ for neural network API support
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-alpha"

        ndk {
            // S25 Ultra uses Snapdragon 8 Elite (ARM v9)
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildFeatures {
        compose = true
        mlModelBinding = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.15"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // Android core
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")

    // Compose UI
    implementation(platform("androidx.compose:compose-bom:2024.11.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")

    // On-device ML / Neural Networks
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("com.google.ai.edge.litert:litert-support-api:1.0.1")

    // Samsung Neural SDK (for Snapdragon 8 Elite NPU direct access)
    // implementation("com.samsung.android:sdk-neural:1.0.0")

    // Qualcomm AI Engine Direct (QNN) - for Snapdragon NPU
    // implementation("com.qualcomm.qti:qnn-runtime:2.25.0")

    // CameraX for multi-camera sensor fusion
    implementation("androidx.camera:camera-core:1.4.0")
    implementation("androidx.camera:camera-camera2:1.4.0")
    implementation("androidx.camera:camera-lifecycle:1.4.0")

    // Room for persistent memory
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    annotationProcessor("androidx.room:room-compiler:2.6.1")

    // Coroutines for async sensor processing
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // DataStore for fast key-value state
    implementation("androidx.datastore:datastore-preferences:1.1.1")
}
