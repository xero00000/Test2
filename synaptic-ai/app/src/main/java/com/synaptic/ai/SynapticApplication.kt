package com.synaptic.ai

import android.app.Application
import com.synaptic.ai.runtime.InferenceRuntime

class SynapticApplication : Application() {

    lateinit var runtime: InferenceRuntime
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this
        runtime = InferenceRuntime(this)
        runtime.initialize()
    }

    override fun onTerminate() {
        runtime.stop()
        super.onTerminate()
    }

    companion object {
        lateinit var instance: SynapticApplication
            private set
    }
}
