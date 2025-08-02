# TTS-italiano
Addestramento di un AI generativa TTS in italiano

``` python
python train_speecht5.py # Downloads data, processes everything, saves to cache
```
Questa esecuzione fa caching di vari elementi, per cui puo' essere utile anche forzare il ricalcolo:

``` python
python train_speecht5.py --force-reprocess # Ignores cache, reprocesses everything
```

``` python
python train_speecht5.py --clear-cache # Deletes cache files
```