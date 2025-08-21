# variables.tf - CPU Instance Variables

# Variabili di progetto e regione
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
# Variabili istanza
variable "cpu_instance_name" {
  description = "Name of the CPU instance"
  type        = string
  default     = "pre-training"
}

variable "machine_type" {
  description = "Machine type for the CPU instance"
  type        = string
  default     = "e2-standard-4"
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

# Variabili disco boot
variable "boot_image" {
  description = "Boot disk image"
  type        = string
  default     = "projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2404-noble-amd64-v20250725"
}

variable "boot_disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

variable "boot_disk_type" {
  description = "Boot disk type"
  type        = string
  default     = "pd-balanced"
}

# Variabili disco cache
variable "enable_cache_disk" {
  description = "Whether to create and attach a cache disk"
  type        = bool
  default     = true
}

variable "cache_disk_name" {
  description = "Name of the cache disk"
  type        = string
  default     = "disk-cache"
}

variable "cache_disk_size" {
  description = "Cache disk size in GB"
  type        = number
  default     = 200
}

variable "cache_disk_mount_point" {
  description = "Mount point for the cache disk"
  type        = string
  default     = "/mnt/cache"
}

variable "cache_disk_type" {
  description = "Cache disk type"
  type        = string
  default     = "pd-balanced"
}

# Variabili scheduling
variable "on_host_maintenance" {
  description = "Behavior during host maintenance (MIGRATE for CPU, TERMINATE for GPU)"
  type        = string
  default     = "MIGRATE"
}

variable "preemptible" {
  description = "Whether the instance should be preemptible"
  type        = bool
  default     = false
}

variable "provisioning_model" {
  description = "Provisioning model (STANDARD or SPOT)"
  type        = string
  default     = "STANDARD"
}

variable "compute_service_account_email" {
  description = "Email del service account per l'istanza compute"
  type        = string
  default     = "792353693062-compute@developer.gserviceaccount.com"
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
