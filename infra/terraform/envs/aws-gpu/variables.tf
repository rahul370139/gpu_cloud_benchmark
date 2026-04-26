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

variable "controller_ami_id" {
  type = string
}

variable "worker_ami_id" {
  type    = string
  default = null
}

variable "controller_instance_type" {
  type    = string
  default = "t3.large"
}

variable "controller_hourly_rate" {
  type    = number
  default = 0.0832
}

variable "worker_pools" {
  description = "List of GPU worker pools; one Kubernetes benchmark Job is run per pool."
  type = list(object({
    gpu_class     = string
    instance_type = string
    hourly_rate   = number
    count         = number
  }))
  default = [
    {
      gpu_class     = "T4"
      instance_type = "g4dn.xlarge"
      hourly_rate   = 0.526
      count         = 1
    },
  ]
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
