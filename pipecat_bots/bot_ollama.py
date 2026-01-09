#!/usr/bin/env python3
#
# Pipecat bot using Ollama for LLM inference.
#
# Designed for running on hardware with limited VRAM (e.g., RTX 3070 with 8GB).
# Uses Ollama's OpenAI-compatible API with a small model like Llama 3.2 3B.
#
# Environment variables:
#   OLLAMA_URL            Ollama API URL (default: http://localhost:11434/v1)
#   OLLAMA_MODEL          Model name (default: llama3.2:3b)
#   NVIDIA_ASR_URL        ASR WebSocket URL (default: ws://localhost:8080)
#   NVIDIA_TTS_URL        Magpie TTS server URL (default: http://localhost:8001)
#
# Usage:
#   uv run pipecat_bots/bot_ollama.py
#   uv run pipecat_bots/bot_ollama.py -t daily
#   uv run pipecat_bots/bot_ollama.py -t webrtc
#
# To run LLM-only (text mode, no ASR/TTS):
#   NVIDIA_ASR_URL="" NVIDIA_TTS_URL="" uv run pipecat_bots/bot_ollama.py
#

import os

from dotenv import load_dotenv
from loguru import logger
from magpie_websocket_tts import MagpieWebSocketTTSService

# Import our custom local services
from nvidia_stt import NVidiaWebSocketSTTService
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from sentence_aggregator import SentenceAggregator
from v2v_metrics import V2VMetricsProcessor

load_dotenv(override=True)

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ASR/TTS configuration (can be disabled by setting to empty string)
NVIDIA_ASR_URL = os.getenv("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_TTS_URL = os.getenv("NVIDIA_TTS_URL", "http://localhost:8001")

# VAD configuration
VAD_STOP_SECS = 0.2

# Transport configurations with VAD and SmartTurn analyzer
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Ollama bot")
    logger.info(f"  Ollama URL: {OLLAMA_URL}")
    logger.info(f"  Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"  ASR URL: {NVIDIA_ASR_URL or 'DISABLED'}")
    logger.info(f"  TTS URL: {NVIDIA_TTS_URL or 'DISABLED'}")
    logger.info(f"  Transport: {type(transport).__name__}")
    logger.info(f"  VAD stop_secs: {VAD_STOP_SECS}s")

    # Build pipeline processors list
    pipeline_processors = [transport.input()]

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    pipeline_processors.append(rtvi)

    # ASR (optional)
    if NVIDIA_ASR_URL:
        stt = NVidiaWebSocketSTTService(
            url=NVIDIA_ASR_URL,
            sample_rate=16000,
        )
        pipeline_processors.append(stt)
        logger.info("ASR enabled")
    else:
        logger.info("ASR disabled - text input only")

    # Context aggregator (user side)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Your goal is to have a natural conversation with the user. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Avoid special characters. Use only simple, plain text sentences. "
                "Always punctuate your responses using standard sentence punctuation: "
                "commas, periods, question marks, exclamation points, etc. "
                "Always spell out numbers as words."
            ),
        },
        {
            "role": "user",
            "content": "Say hello and ask how you can help.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    pipeline_processors.append(context_aggregator.user())

    # Ollama LLM via OpenAI-compatible API
    llm = OpenAILLMService(
        api_key="ollama",  # Ollama doesn't need a real key
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
    )
    pipeline_processors.append(llm)
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")

    # Sentence aggregator for streaming to TTS
    sentence_aggregator = SentenceAggregator()
    pipeline_processors.append(sentence_aggregator)

    # TTS (optional)
    if NVIDIA_TTS_URL:
        tts = MagpieWebSocketTTSService(
            server_url=NVIDIA_TTS_URL,
            voice="aria",
            language="en",
            params=MagpieWebSocketTTSService.InputParams(
                language="en",
                streaming_preset="conservative",
                use_adaptive_mode=True,
            ),
        )
        pipeline_processors.append(tts)
        logger.info("TTS enabled (adaptive mode)")
    else:
        logger.info("TTS disabled - text output only")

    # Voice-to-voice metrics (only meaningful with ASR+TTS)
    if NVIDIA_ASR_URL and NVIDIA_TTS_URL:
        v2v_metrics = V2VMetricsProcessor(vad_stop_secs=VAD_STOP_SECS)
        pipeline_processors.append(v2v_metrics)

    pipeline_processors.append(transport.output())
    pipeline_processors.append(context_aggregator.assistant())

    pipeline = Pipeline(pipeline_processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        await rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
