import tortoise.api as tortoise_api
from tortoise.utils.audio import load_voice

# Carica Tortoise
tts = tortoise_api.TextToSpeech()

# Carica una voce italiana
voice_samples, conditioning_latents = load_voice("tortoise_italian/tortoise_voices/speaker_123")

# Genera audio italiano
text = "Ciao! Come stai oggi? La mia pronuncia italiana è molto naturale."
gen = tts.tts_with_preset(
    text, 
    voice_samples=voice_samples, 
    conditioning_latents=conditioning_latents, 
    preset='high_quality'  # o 'fast' per velocità
)

# Salva
import torchaudio
torchaudio.save("output_italiano.wav", gen.squeeze(0).cpu(), 24000)