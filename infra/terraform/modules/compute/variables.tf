variable "name" {
  type = string
}

variable "region" {
  type = string
}

variable "controller_ami_id" {
  type = string
}

variable "worker_ami_id" {
  type = string
}

variable "controller_instance_type" {
  type = string
}

variable "controller_hourly_rate" {
  type = number
}

variable "worker_pools" {
  description = <<EOT
List of worker pools, one per GPU class. Each pool spins up `count`
identical EC2 instances tagged and K8s-labeled with `gpu_class` so the
benchmark Job can target them via nodeSelector.
EOT
  type = list(object({
    gpu_class     = string
    instance_type = string
    hourly_rate   = number
    count         = number
  }))

  validation {
    condition     = length(var.worker_pools) > 0
    error_message = "worker_pools must contain at least one pool."
  }
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "key_name" {
  type = string
}

variable "cluster_token" {
  type      = string
  sensitive = true
}

variable "artifact_bucket_name" {
  type = string
}

variable "force_destroy_bucket" {
  type    = bool
  default = true
}

variable "root_volume_size" {
  type    = number
  default = 200
}

variable "benchmark_run_id" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}
