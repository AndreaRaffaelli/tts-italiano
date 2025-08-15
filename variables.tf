# `variables.tf`
# `variables.tf` - L4 GPU Instance Variables

variable "project_id" {
  description = "Google Cloud Project ID"
  default     = "tts-project-andrea"
  type        = string
}

variable "region" {
  description = "La regione in cui creare le risorse"
  type        = string
  default     = "us-central1" # Regione con buona disponibilità di GPU L4
}

variable "zone" {
  description = "La zona specifica all'interno della regione"
  type        = string
  default     = "us-central1-a" # Zona con disponibilità GPU L4 garantita
}

variable "gpu_instance_name" {
  description = "Nome dell'istanza GPU L4"
  type        = string
  default     = "instance-20250802-231725"
}

variable "machine_type" {
  description = "Tipo di macchina per l'istanza GPU"
  type        = string
  default     = "g2-standard-4" # Configurazione ottimizzata per GPU L4
}

variable "gpu_count" {
  description = "Numero di GPU L4 da allocare"
  type        = number
  default     = 1
  validation {
    condition     = var.gpu_count >= 1 && var.gpu_count <= 4
    error_message = "Il numero di GPU deve essere tra 1 e 4."
  }
}

variable "boot_image" {
  description = "Immagine del sistema operativo per il boot disk"
  type        = string
  default     = "projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2404-noble-amd64-v20250725"
}

variable "boot_disk_size" {
  description = "Dimensione del boot disk in GB"
  type        = number
  default     = 300
}

variable "boot_disk_type" {
  description = "Tipo di disco per il boot disk"
  type        = string
  default     = "pd-balanced"
  validation {
    condition     = contains(["pd-standard", "pd-ssd", "pd-balanced"], var.boot_disk_type)
    error_message = "Il tipo di disco deve essere pd-standard, pd-ssd, o pd-balanced."
  }
}

variable "preemptible" {
  description = "Se l'istanza deve essere preemptible (più economica ma può essere terminata)"
  type        = bool
  default     = false
}

variable "provisioning_model" {
  description = "Modello di provisioning dell'istanza"
  type        = string
  default     = "STANDARD"
  validation {
    condition     = contains(["STANDARD", "SPOT"], var.provisioning_model)
    error_message = "Il modello di provisioning deve essere STANDARD o SPOT."
  }
}

variable "deletion_protection" {
  description = "Se abilitare la protezione dalla cancellazione"
  type        = bool
  default     = false
}

variable "compute_service_account_email" {
  description = "Email del service account per l'istanza compute"
  type        = string
  default     = "792353693062-compute@developer.gserviceaccount.com"
}

variable "environment" {
  description = "Environment label per le risorse"
  type        = string
  default     = "development"
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "L'environment deve essere development, staging, o production."
  }
}

# Variabili SSH
variable "ssh_user" {
  description = "Il nome utente SSH per la connessione alla VM"
  type        = string
  default     = "andrea"
}

variable "ssh_public_key_path" {
  description = "Il percorso del file con la tua chiave pubblica SSH (lasciare vuoto se non si vuole configurare SSH)"
  type        = string
  default     = "/home/andrea/.ssh/tts-prova.pub" # Percorso predefinito della chiave pubblica SSH
}
