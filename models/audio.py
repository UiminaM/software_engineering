import torch
from IPython.display import Audio

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils

sampling_rate = 16000 
audio = read_audio('okey.mp3', sampling_rate=sampling_rate)
speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sampling_rate)
print(speech_timestamps)