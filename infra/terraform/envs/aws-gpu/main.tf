terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
    template = {
      source  = "hashicorp/template"
      version = "~> 2.2"
    }
  }
}

provider "aws" {
  region = var.region
}

locals {
  common_tags = merge(var.tags, {
    Project     = "ml-gpu-benchmark"
    ManagedBy   = "terraform"
    Environment = var.environment
    Owner       = "Sahil"
  })
}

module "network" {
  source = "../../modules/network"

  name                = var.name
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidrs = var.public_subnet_cidrs
  availability_zones  = var.availability_zones
  tags                = local.common_tags
}

module "security" {
  source = "../../modules/security"

  name        = var.name
  vpc_id      = module.network.vpc_id
  admin_cidrs = var.admin_cidrs
  tags        = local.common_tags
}

module "compute" {
  source = "../../modules/compute"

  name                     = var.name
  region                   = var.region
  ami_id                   = var.ami_id
  controller_instance_type = var.controller_instance_type
  worker_instance_type     = var.worker_instance_type
  controller_hourly_rate   = var.controller_hourly_rate
  worker_hourly_rate       = var.worker_hourly_rate
  worker_count             = var.worker_count
  subnet_ids               = module.network.public_subnet_ids
  security_group_id        = module.security.security_group_id
  key_name                 = var.key_name
  cluster_token            = var.cluster_token
  artifact_bucket_name     = var.artifact_bucket_name
  benchmark_run_id         = var.benchmark_run_id
  tags                     = local.common_tags
}

resource "local_file" "inventory" {
  filename = "${path.module}/inventory.json"
  content = jsonencode({
    controller_public_ip  = module.compute.controller_public_ip
    controller_private_ip = module.compute.controller_private_ip
    worker_public_ips     = module.compute.worker_public_ips
    artifact_bucket_name  = module.compute.artifact_bucket_name
    benchmark_run_id      = var.benchmark_run_id
    region                = var.region
  })
}
