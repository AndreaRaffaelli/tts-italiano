# Enhanced TTS Training Pipeline

Una pipeline modulare e completa per il fine-tuning di modelli Text-to-Speech italiani usando SpeechT5.

## 🚀 Caratteristiche

- **Modulare**: Architettura ben organizzata con separazione delle responsabilità
- **Cache intelligente**: Sistema di cache avanzato per evitare rielaborazioni
- **GPU ottimizzato**: Rilevamento automatico GPU e ottimizzazione batch size
- **Preprocessing robusto**: Preprocessing avanzato per audio e testo italiano
- **CLI interattiva**: Interfaccia a riga di comando user-friendly
- **Monitoraggio**: Sistema di logging e monitoraggio integrato
- **Hugging Face ready**: Integrazione completa con Hugging Face Hub

## 📋 Requisiti

- Python 3.8+
- CUDA-capable GPU (raccomandato, ma funziona anche su CPU)
- Almeno 16GB di RAM
- Spazio su disco: ~10GB per dataset e cache

## ⚡ Installazione Rapida

```bash
# Clona il repository
git clone https://github.com/yourusername/enhanced-tts-training.git
cd enhanced-tts-training

# Installa dipendenze
pip install -r requirements.txt

# Oppure installa come package
pip install -e .
```

## 🔧 Installazione Dettagliata

### 1. Dipendenze PyTorch (con CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Dipendenze principali
```bash
pip install datasets==3.6.0
pip install soundfile speechbrain accelerate
pip install git+https://github.com/huggingface/transformers.git
```

### 3. Dipendenze opzionali
```bash
# Per il development
pip install -e .[dev]

# Per logging avanzato
pip install -e .[logging]

# Tutto
pip install -e .[all]
```

## 🚀 Utilizzo

### Modo Base
```bash
python main.py
```

### Con parametri personalizzati
```bash
python main.py \
  --output-dir "my_italian_tts" \
  --max-steps 5000 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --cache-dir "my_cache"
```

### Modo non-interattivo
```bash
python main.py --yes --max-steps 10000 --push-to-hub
```

### Parametri disponibili

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--output-dir` | Directory output del modello | `speecht5_italian_enhanced` |
| `--max-steps` | Numero massimo di step di training | `10000` |
| `--learning-rate` | Learning rate | `3e-6` |
| `--batch-size` | Override batch size automatico | Auto-detect |
| `--cache-dir` | Directory per i file di cache | `cache` |
| `--force-reprocess` | Forza rielaborazione dati cached | False |
| `--clear-cache` | Pulisce tutta la cache | False |
| `--push-to-hub` | Push su Hugging Face Hub | False |
| `-y, --yes` | Modalità non-interattiva | False |
| `--verbose` | Logging verboso | False |

## 📁 Struttura del Progetto

```
enhanced_tts_training/
├── main.py                    # Entry point principale
├── config/
│   ├── __init__.py
│   └── training_config.py     # Configurazioni di training
├── data/
│   ├── __init__.py
│   ├── preprocessing.py       # Preprocessing audio e testo
│   ├── dataset_loader.py      # Caricamento e gestione dataset
│   └── data_collator.py       # Data collator per il training
├── models/
│   ├── __init__.py
│   └── model_setup.py         # Setup modelli e processori
├── training/
│   ├── __init__.py
│   └── trainer.py             # Logic di training
├── utils/
│   ├── __init__.py
│   ├── cli_utils.py           # Utilities CLI
│   ├── cache_utils.py         # Gestione cache
│   └── gpu_utils.py           # Utilities GPU
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # Questa documentazione
```

## 🔄 Pipeline di Training

La pipeline segue questi passi principali:

1. **Check Dependencies**: Verifica installazione dipendenze
2. **Model Loading**: Carica SpeechT5 e modello speaker embedding
3. **Dataset Loading**: Carica VoxPopuli e Common Voice italiani
4. **Text Preprocessing**: Normalizza testo italiano
5. **Quality Filtering**: Filtra campioni di bassa qualità
6. **Audio Processing**: Preprocessing audio e generazione embedding
7. **Training Setup**: Configura trainer e parametri
8. **Training**: Esegue il fine-tuning
9. **Saving**: Salva modello e processor

## 🎛️ Configurazione Avanzata

### Configurazione da file
```python
from config.training_config import EnhancedTrainingConfig

# Crea configurazione personalizzata
config = EnhancedTrainingConfig(
    max_steps=15000,
    learning_rate=2e-6,
    warmup_steps=2000,
    eval_steps=250,
    save_steps=500
)

# Salva configurazione
config.save_to_file("my_config.json")
```

### Utilizzo programmatico
```python
from training.trainer import create_trainer
from utils.gpu_utils import check_gpu

# Crea trainer
has_gpu = check_gpu()
trainer = create_trainer(args, has_gpu)

# Esegui training
success = trainer.run_training_pipeline()
```

## 📊 Monitoraggio

### TensorBoard
```bash
# Avvia TensorBoard
tensorboard --logdir speecht5_italian_enhanced/logs

# Oppure se hai specificato una directory diversa
tensorboard --logdir your_output_dir/logs
```

### Logs personalizzati
Il sistema genera automaticamente:
- Statistiche dataset
- Metriche di training
- Uso memoria GPU
- Curve di loss

## 🔧 Risoluzione Problemi

### Out of Memory (OOM)
```bash
# Riduci batch size
python main.py --batch-size 1

# Oppure usa gradient checkpointing (già abilitato di default)
```

### Cache corrotta
```bash
# Pulisci cache
python main.py --clear-cache

# Oppure forza rielaborazione
python main.py --force-reprocess
```

### Datasets non scaricabili
Se i dataset non si scaricano:
1. Verifica connessione internet
2. Controlla spazio disco
3. Prova a scaricare manualmente

## 🚀 Esempi Avanzati

### Training con configurazione personalizzata
```bash
python main.py \
  --output-dir "italian_tts_v2" \
  --max-steps 20000 \
  --learning-rate 1e-5 \
  --cache-dir "/tmp/tts_cache" \
  --push-to-hub \
  --yes
```

### Solo preprocessing (senza training)
```python
from data.dataset_loader import DatasetManager
from utils.cache_utils import get_cache_paths

cache_paths = get_cache_paths("my_cache")
manager = DatasetManager(cache_paths, processor, speaker_model)
dataset = manager.load_and_process_datasets()
```

## 🤝 Contribuire

1. Fork il repository
2. Crea un branch per la feature (`git checkout -b feature/nuova-feature`)
3. Commit le modifiche (`git commit -am 'Aggiunge nuova feature'`)
4. Push al branch (`git push origin feature/nuova-feature`)
5. Crea una Pull Request

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi il file `LICENSE` per dettagli.

## 🙏 Ringraziamenti

- [Hugging Face](https://huggingface.co/) per Transformers e Datasets
- [SpeechBrain](https://speechbrain.github.io/) per i modelli audio
- [Microsoft](https://github.com/microsoft/SpeechT5) per SpeechT5
- Community italiana di ML per feedback e test

## 📞 Supporto

- Issues: [GitHub Issues](https://github.com/yourusername/enhanced-tts-training/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/enhanced-tts-training/discussions)
- Email: your.email@example.com

---

**Nota**: Questo è un progetto in sviluppo attivo. Le API possono cambiare tra le versioni.
