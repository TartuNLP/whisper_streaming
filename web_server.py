import io
import base64
import librosa
import logging
import argparse

import numpy as np
import soundfile as sf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from whisper_online import asr_factory, set_logging, VACOnlineASRProcessor

listeners = []

# app = FastAPI()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

asr = None
online: VACOnlineASRProcessor
in_use = False


# Copied from https://stackoverflow.com/a/16891418
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def seconds_to_timecode(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    else:
        return f"{minutes:02d}:{seconds:05.2f}"

def init_asr(model_dir, language):
    global asr, online
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--lan', type=str, default=language)
    parser.add_argument('--model', type=str, default='large-v3-turbo')
    parser.add_argument('--model_cache_dir', type=str, default=None)
    parser.add_argument('--task', type=str, default='transcribe')
    parser.add_argument('--vad', action='store_true', default=True)
    parser.add_argument('--vac', action='store_true', default=True)
    parser.add_argument('--vac-chunk-size', type=float, default=5.0)
    parser.add_argument('--backend', type=str, default='faster-whisper')
    parser.add_argument('--buffer_trimming', type=str, default='segment')
    parser.add_argument('--buffer_trimming_sec', type=float, default=20.0)
    parser.add_argument('--min-chunk-size', type=float, default=1.0)
    parser.add_argument('--log-level', type=str, default='CRITICAL')
    
    args = parser.parse_args([])
    set_logging(args, logger)
    
    asr, online = asr_factory(args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_asr(model_dir="/raid/rauno/whisper-large-v3-turbo-et-subs/ct2/", language="et")
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/check_availability")
async def check_availability():
    global in_use
    if in_use:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        return status.HTTP_200_OK

@app.websocket("/listen")
async def listener_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Listener open")
    listeners.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Listener closed")
        listeners.remove(websocket)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global in_use
    if in_use:
        await websocket.close(code=503, reason="Server overloaded")
        return
    
    try:
        in_use = True
        await websocket.accept()
        logger.info(f"Connection open")
        online.init()

        while True:
            try:
                data = await websocket.receive_text()

                if not data:
                    continue

                if data.startswith("$translation"):
                    if listeners:
                        text = remove_prefix(data, "$translation")
                        newline = False
                        if text.startswith("$newline"):
                            text = remove_prefix(text, "$newline")
                            newline = True
                        for child in listeners:
                            await child.send_json({"translation": True, "newline": newline, "text": text})
                    continue
                    
                try:
                    audio_data = base64.b64decode(data.split(',')[1])
                    
                    with io.BytesIO(audio_data) as buffer:
                        audio_array, sample_rate = sf.read(buffer)
                        logger.debug(f"Loaded audio chunk: shape={audio_array.shape}, sample_rate={sample_rate}")
                        
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.mean(axis=1)
                        
                        audio_array = audio_array.astype(np.float32)
                        if np.abs(audio_array).max() > 1.0:
                            audio_array = audio_array / np.abs(audio_array).max()

                        if sample_rate != 16000:
                            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                            logger.debug("Resampled audio chunk to 16kHz")

                        online.insert_audio_chunk(audio_array)
                        result = online.process_iter()

                        if result[0] is not None:
                            start_tc = seconds_to_timecode(result[0])
                            end_tc = seconds_to_timecode(result[1])
                            
                            response = {
                                'start': float(result[0]),
                                'end': float(result[1]),
                                'start_tc': start_tc,                                
                                'end_tc': end_tc,
                                'text': result[2],
                            }
                            await websocket.send_json(response)

                            for child in listeners:
                                await child.send_json(response)
                        
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await websocket.send_json({"error": f"Audio processing error: {str(e)}"})
                    
            except Exception as e:
                logger.error(f"WebSocket receive error: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        try:
            final_result = online.finish()
            if final_result[0] is not None:
                start_tc = seconds_to_timecode(final_result[0])
                end_tc = seconds_to_timecode(final_result[1])
                
                await websocket.send_json({
                    'start': float(final_result[0]),
                    'end': float(final_result[1]),
                    'start_tc': start_tc,
                    'end_tc': end_tc,
                    'text': final_result[2],
                    'final': True
                })
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
        logger.info(f"Connection closed")
        for child_connection in listeners:
            await child_connection.close()
        in_use = False
