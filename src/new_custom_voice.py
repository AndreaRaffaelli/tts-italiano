# Con SpeechT5 + SpeechBrain
from speechbrain.inference.classifiers import EncoderClassifier

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb"
)

# Da file audio
import torchaudio
waveform, sr = torchaudio.load("mia_voce.wav")
waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

# Estrai embedding
with torch.no_grad():
    embedding = speaker_model.encode_batch(waveform)
    embedding = torch.nn.functional.normalize(embedding, dim=2)

# Salva
torch.save(embedding, "mio_speaker_embedding.pt")