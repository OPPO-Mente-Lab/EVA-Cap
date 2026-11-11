# Modified by OPPO in 2026 from Qwen2.5-Omni qwen-omni-utils to support
# additional audio input sources.

import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np

import requests
import tempfile
import os
import subprocess


SAMPLE_RATE=16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def load_audio_from_any_source(data, audio_start=0.0, audio_end=None, sample_rate=16000):
    """
    支持高清音频解码，兼容 mp3, mp4, webm, flac, base64, BytesIO, URL, local path
    返回 (numpy waveform,)
    """
    if isinstance(data, np.ndarray):
        return data[int(audio_start * sample_rate): None if audio_end is None else int(audio_end * sample_rate)]

    if isinstance(data, BytesIO):
        # Bytes —> 临时文件
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(data.read())
            tmp.flush()
            tmp_path = tmp.name
    elif isinstance(data, str):
        if data.startswith("http://") or data.startswith("https://"):
            resp = requests.get(data)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
                tmp.write(resp.content)
                tmp.flush()
                tmp_path = tmp.name
        elif os.path.exists(data):
            tmp_path = data
        else:
            raise ValueError(f"Invalid path or data: {data}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # 使用 ffmpeg 提取音频，输出 bytes 流 => librosa 支持的格式
    cmd = [
        "ffmpeg", "-ss", str(audio_start),  # 起始
        "-i", tmp_path,
        "-t", str(audio_end - audio_start) if audio_end else "9999",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),  # 采样率
        "-ac", "1",               # 单通道
        "-"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        audio, _ = librosa.load(BytesIO(out), sr=sample_rate)
        return audio
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e}")
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path) and ".tmp" in tmp_path:
            os.remove(tmp_path)

def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            #data = audioread.ffdec.FFmpegAudioFile(path)
                            response = requests.get(path)
                            response.raise_for_status()  # 防止网络失败
                            data = BytesIO(response.content)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            #data = audioread.ffdec.FFmpegAudioFile(path)
                            response = requests.get(path)
                            response.raise_for_status()  # 防止网络失败
                            data = BytesIO(response.content)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                '''
                audios.append(
                    librosa.load(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None,
                    )[0]
                )
                '''
                audios.append(
                    load_audio_from_any_source(
                        data,
                        audio_start=audio_start,
                        audio_end=audio_end,
                        sample_rate=SAMPLE_RATE
                    )
                )
    if len(audios) == 0:
        audios = None
    return audios
