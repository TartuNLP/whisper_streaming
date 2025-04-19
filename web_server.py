from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import soundfile as sf
import io
import base64
import json
import asyncio
import logging
import librosa
from whisper_online import asr_factory, set_logging, OnlineASRProcessor
import argparse

def seconds_to_timecode(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    else:
        return f"{minutes:02d}:{seconds:05.2f}"

app = FastAPI()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

asr = None
online = None

def init_asr(model_dir, language):
    global asr, online
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--lan', type=str, default=language)
    parser.add_argument('--model', type=str, default='large-v3-turbo')
    parser.add_argument('--model_cache_dir', type=str, default=model_dir)
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

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("connection open")
    
    try:
        online.init()
        
        while True:
            try:
                data = await websocket.receive_text()

                if not data:
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
        logger.info("connection closed")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        audio_data = await file.read()
        
        with io.BytesIO(audio_data) as buffer:
            audio_array, sample_rate = sf.read(buffer)
            logger.debug(f"Loaded audio: shape={audio_array.shape}, sample_rate={sample_rate}")
            
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()

            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                logger.debug("Resampled audio to 16kHz")

        logger.debug("Starting transcription")
        result = asr.transcribe(audio_array)
        logger.debug(f"Transcription complete: {type(result)}")

        transcription_segments = []
        

        for segment in result:
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'start_tc': seconds_to_timecode(segment.start),
                'end_tc': seconds_to_timecode(segment.end)
            })

        if not transcription_segments:
            logger.warning("No transcription segments were generated")
        else:
            logger.info(f"Generated {len(transcription_segments)} segments")
            
        return JSONResponse(content={"segments": transcription_segments})
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    init_asr(model_dir="model/", language="et")
    uvicorn.run(app, host="0.0.0.0", port=8000)