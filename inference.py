
import os
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

MODEL_PATH = "path/to/EVA-CAP"
VIDEO_PATH = "path/to/video.mp4"
USE_AUDIO_IN_VIDEO = True
VIDEO_MAX_PIXELS = 401408  # 512 * 28 * 28
os.environ["VIDEO_MAX_PIXELS"] = str(512 * 28 * 28 * 50)

PROMPT = "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so."

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH, "max_pixels": VIDEO_MAX_PIXELS},
            {"type": "text", "text": PROMPT},
        ],
    },
]

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
inputs = inputs.to(model.device).to(model.dtype)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        do_sample=False,
        max_new_tokens=2048,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        repetition_penalty=1.1,
    )

caption = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
caption = caption.split("\nassistant\n")[-1]
print(caption)
