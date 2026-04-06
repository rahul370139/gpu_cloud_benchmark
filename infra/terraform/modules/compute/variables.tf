variable "name" {
  type = string
}

variable "region" {
  type = string
}

variable "ami_id" {
  type = string
}

variable "controller_instance_type" {
  type = string
}

variable "worker_instance_type" {
  type = string
}

variable "controller_hourly_rate" {
  type = number
}

variable "worker_hourly_rate" {
  type = number
}

variable "worker_count" {
  type = number
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
  type = string
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
