# TTS-italiano
Addestramento di un AI generativa TTS in italiano

``` python
python train_speecht5.py # Downloads data, processes everything, saves to cache
```
Esecuzione non interattiva:

``` python
python train_speecht5.py --yes # Non interactive 
```
Questa esecuzione fa caching di vari elementi, per cui puo' essere utile anche forzare il ricalcolo:

``` python
python train_speecht5.py --force-reprocess # Ignores cache, reprocesses everything
```

``` python
python train_speecht5.py --clear-cache # Deletes cache files
```

## Enanched version

La prima dava risultati un po' grezzi. Riproviamo:
### Training normale con conferme
python enhanced_tts_training.py

### Training automatico senza conferme
python enhanced_tts_training.py --yes

### Forza riprocessing (ignora cache)
python enhanced_tts_training.py --force-reprocess --yes

### Pulisci cache
python enhanced_tts_training.py --clear-cache

### Training con parametri custom
python enhanced_tts_training.py --yes --max-steps 15000 --output-dir "my_italian_tts" --push-to-hub

### Usa directory cache custom
python enhanced_tts_training.py --cache-dir "/mnt/fast_ssd/cache" --yes