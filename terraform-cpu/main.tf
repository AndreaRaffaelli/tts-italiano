# `main.tf` - CPU Instance Configuration
# Configurazione del provider Google
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.74.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 4.74.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Risorsa per la creazione dell'istanza CPU
resource "google_compute_instance" "cpu_instance" {
  name         = var.cpu_instance_name
  machine_type = var.machine_type
  zone         = var.zone

  # Disco aggiuntivo per cache (opzionale)
  dynamic "attached_disk" {
    for_each = var.enable_cache_disk ? [1] : []
    content {
      device_name = var.cache_disk_name
      mode        = "READ_WRITE"
      source      = google_compute_disk.cache_disk[0].self_link
    }
  }

  boot_disk {
    auto_delete = true
    device_name = var.cpu_instance_name
    
    initialize_params {
      image = var.boot_image
      size  = var.boot_disk_size
      type  = var.boot_disk_type
    }
    
    mode = "READ_WRITE"
  }

  network_interface {
    subnetwork = "projects/${var.project_id}/regions/${var.region}/subnetworks/default"
    
    access_config {
      network_tier = "PREMIUM"
    }
    
    queue_count = 0
    stack_type  = "IPV4_ONLY"
  }

  # Configurazioni di sicurezza e monitoraggio
  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  # Configurazione scheduling per CPU (diversa da GPU)
  scheduling {
    automatic_restart   = true
    on_host_maintenance = var.on_host_maintenance  # MIGRATE per CPU, TERMINATE per GPU
    preemptible         = var.preemptible
    provisioning_model  = var.provisioning_model
  }

  # Service account con permessi necessari
  service_account {
    email = var.compute_service_account_email
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append"
    ]
  }

  # Metadati per SSH e configurazioni
 metadata = {
    ssh-keys               = var.ssh_public_key_path != "" ? "${var.ssh_user}:${file(var.ssh_public_key_path)}" : null
    enable-osconfig        = "TRUE"
    # metadata-startup-script = var.enable_cache_disk ? local.startup_script : null
  }

  # Labels per organizzazione e policy
  labels = {
    goog-ec-src           = "vm_add-tf"
    goog-ops-agent-policy = "v2-x86-template-1-4-0"
    environment           = var.environment
    instance-type         = "cpu"
  }

  can_ip_forward      = false
  enable_display      = false
}

# Disco aggiuntivo opzionale per cache
resource "google_compute_disk" "cache_disk" {
  count = var.enable_cache_disk ? 1 : 0
  
  name  = var.cache_disk_name
  type  = var.cache_disk_type
  zone  = var.zone
  size  = var.cache_disk_size
  
  labels = {
    environment = var.environment
    purpose     = "cache"
  }

  # Prevent accidental destruction of the cache disk
  lifecycle {
    prevent_destroy = false
  }
}

# Modulo per Ops Agent Policy (per monitoraggio)
module "ops_agent_policy" {
  source        = "github.com/terraform-google-modules/terraform-google-cloud-operations/modules/ops-agent-policy"
  project       = var.project_id
  zone          = var.zone
  assignment_id = "goog-ops-agent-v2-x86-template-1-4-0-${var.zone}-${var.cpu_instance_name}"
  
  agents_rule = {
    package_state = "installed"
    version       = "latest"
  }
  
  instance_filter = {
    all = false
    inclusion_labels = [{
      labels = {
        goog-ops-agent-policy = "v2-x86-template-1-4-0"
      }
    }]
  }
}

# Output per IP esterno istanza CPU
output "cpu_instance_external_ip" {
  value       = google_compute_instance.cpu_instance.network_interface[0].access_config[0].nat_ip
  description = "The external IP address of the CPU instance"
}

# Output per IP interno istanza CPU
output "cpu_instance_internal_ip" {
  value       = google_compute_instance.cpu_instance.network_interface[0].network_ip
  description = "The internal IP address of the CPU instance"
}

# Output per nome istanza
output "cpu_instance_name" {
  value       = google_compute_instance.cpu_instance.name
  description = "The name of the CPU instance"
}

# Output per self link
output "cpu_instance_self_link" {
  value       = google_compute_instance.cpu_instance.self_link
  description = "The self link of the CPU instance"
}

# Output per disco cache (se abilitato)
output "cache_disk_self_link" {
  value       = var.enable_cache_disk ? google_compute_disk.cache_disk[0].self_link : null
  description = "The self link of the cache disk (if enabled)"
}


# # Script di startup per montaggio automatico disco cache
# locals {
#   startup_script = <<-EOF
# #!/bin/bash
# set -e

# # Log di avvio
# echo "$(date): Avvio script di configurazione disco cache" >> /var/log/disk-setup.log

# # Funzione per logging
# log() {
#     echo "$(date): $1" >> /var/log/disk-setup.log
#     echo "$1"
# }

# # Verifica se il disco cache esiste
# CACHE_DEVICE="/dev/disk/by-id/google-${var.cache_disk_name}"
# MOUNT_POINT="${var.cache_disk_mount_point}"

# log "Ricerca disco cache: $CACHE_DEVICE"

# # Attendi che il disco sia disponibile (max 60 secondi)
# for i in {1..60}; do
#     if [ -L "$CACHE_DEVICE" ]; then
#         log "Disco cache trovato: $CACHE_DEVICE"
#         break
#     fi
#     if [ $i -eq 60 ]; then
#         log "ERRORE: Disco cache non trovato dopo 60 secondi"
#         exit 1
#     fi
#     sleep 1
# done

# # Risolve il link simbolico per ottenere il dispositivo reale
# REAL_DEVICE=$(readlink -f "$CACHE_DEVICE")
# log "Dispositivo reale: $REAL_DEVICE"

# # Verifica se il disco ha già un filesystem
# if ! blkid "$REAL_DEVICE" > /dev/null 2>&1; then
#     log "Formattazione del disco cache con ext4..."
#     mkfs.ext4 -F "$REAL_DEVICE"
#     log "Formattazione completata"
# else
#     log "Disco cache già formattato"
# fi

# # Crea il punto di montaggio se non esiste
# if [ ! -d "$MOUNT_POINT" ]; then
#     log "Creazione directory di montaggio: $MOUNT_POINT"
#     mkdir -p "$MOUNT_POINT"
# fi

# # Verifica se è già montato
# if ! mountpoint -q "$MOUNT_POINT"; then
#     log "Montaggio del disco cache in $MOUNT_POINT"
#     mount "$REAL_DEVICE" "$MOUNT_POINT"
#     log "Montaggio completato"
# else
#     log "Disco cache già montato in $MOUNT_POINT"
# fi

# # Ottieni UUID del disco
# DISK_UUID=$(blkid -s UUID -o value "$REAL_DEVICE")
# log "UUID del disco: $DISK_UUID"

# # Aggiungi o aggiorna entry in /etc/fstab per montaggio automatico
# FSTAB_ENTRY="UUID=$DISK_UUID $MOUNT_POINT ext4 defaults,nofail 0 2"

# # Rimuovi eventuali entry esistenti per questo mount point
# grep -v "$MOUNT_POINT" /etc/fstab > /tmp/fstab.new || true

# # Aggiungi la nuova entry
# echo "$FSTAB_ENTRY" >> /tmp/fstab.new

# # Sostituisci /etc/fstab
# mv /tmp/fstab.new /etc/fstab

# log "Entry fstab aggiunta: $FSTAB_ENTRY"

# # Imposta permessi appropriati
# chown ${var.ssh_user}:${var.ssh_user} "$MOUNT_POINT" 2>/dev/null || chown ubuntu:ubuntu "$MOUNT_POINT"
# chmod 755 "$MOUNT_POINT"

# log "Permessi impostati per $MOUNT_POINT"

# # Verifica finale
# if df -h | grep -q "$MOUNT_POINT"; then
#     DISK_INFO=$(df -h | grep "$MOUNT_POINT")
#     log "✅ Configurazione disco cache completata con successo!"
#     log "📊 Informazioni disco: $DISK_INFO"
# else
#     log "❌ ERRORE: Il disco non risulta montato correttamente"
#     exit 1
# fi

# log "Script di configurazione disco completato"
# EOF
# }
