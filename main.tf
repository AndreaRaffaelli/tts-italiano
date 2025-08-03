# `main.tf`
# `main.tf` - L4 GPU Instance Configuration
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

# Risorsa per la creazione dell'istanza GPU L4
resource "google_compute_instance" "gpu_l4_instance" {
  name         = var.gpu_instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    auto_delete = true
    device_name = var.gpu_instance_name
    
    initialize_params {
      image = var.boot_image
      size  = var.boot_disk_size
      type  = var.boot_disk_type
    }
    
    mode = "READ_WRITE"
  }

  # Configurazione GPU L4
  guest_accelerator {
    count = var.gpu_count
    type  = "projects/${var.project_id}/zones/${var.zone}/acceleratorTypes/nvidia-l4"
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

  # Configurazione scheduling per GPU
  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"  # Richiesto per istanze GPU
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
    ssh-keys        = var.ssh_public_key_path != "" ? "${var.ssh_user}:${file(var.ssh_public_key_path)}" : null
    enable-osconfig = "TRUE"
  }

  # Labels per organizzazione e policy
  labels = {
    goog-ec-src           = "vm_add-tf"
    goog-ops-agent-policy = "v2-x86-template-1-4-0"
    environment           = var.environment
    gpu-type              = "nvidia-l4"
  }

  can_ip_forward      = false
  deletion_protection = var.deletion_protection
  enable_display      = false
}

# Modulo per Ops Agent Policy (per monitoraggio) - Commentato per evitare conflitti
# module "ops_agent_policy" {
#   source        = "github.com/terraform-google-modules/terraform-google-cloud-operations/modules/ops-agent-policy"
#   project       = var.project_id
#   zone          = var.zone
#   assignment_id = "goog-ops-agent-v2-x86-template-1-4-0-${var.zone}-${var.gpu_instance_name}"
#   
#   agents_rule = {
#     package_state = "installed"
#     version       = "latest"
#   }
#   
#   instance_filter = {
#     all = false
#     inclusion_labels = [{
#       labels = {
#         goog-ops-agent-policy = "v2-x86-template-1-4-0"
#       }
#     }]
#   }
# }

# Output per IP esterno istanza GPU
output "gpu_instance_external_ip" {
  value       = google_compute_instance.gpu_l4_instance.network_interface[0].access_config[0].nat_ip
  description = "The external IP address of the GPU L4 instance"
}

# Output per IP interno istanza GPU
output "gpu_instance_internal_ip" {
  value       = google_compute_instance.gpu_l4_instance.network_interface[0].network_ip
  description = "The internal IP address of the GPU L4 instance"
}

# Output per nome istanza
output "gpu_instance_name" {
  value       = google_compute_instance.gpu_l4_instance.name
  description = "The name of the GPU L4 instance"
}

# Output per self link
output "gpu_instance_self_link" {
  value       = google_compute_instance.gpu_l4_instance.self_link
  description = "The self link of the GPU L4 instance"
}