variable "name" {
  type    = string
  default = "ml-gpu-bench"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  type = list(string)
}

variable "availability_zones" {
  type = list(string)
}

variable "admin_cidrs" {
  type = list(string)
}

variable "ami_id" {
  type = string
}

variable "controller_instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

variable "worker_instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

variable "controller_hourly_rate" {
  type    = number
  default = 0.526
}

variable "worker_hourly_rate" {
  type    = number
  default = 0.526
}

variable "worker_count" {
  type    = number
  default = 1
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

variable "benchmark_run_id" {
  type    = string
  default = "manual-run"
}

variable "tags" {
  type    = map(string)
  default = {}
}
