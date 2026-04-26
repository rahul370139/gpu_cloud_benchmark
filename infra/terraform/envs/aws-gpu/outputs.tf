output "controller_public_ip" {
  value = module.compute.controller_public_ip
}

output "worker_public_ips" {
  value = module.compute.worker_public_ips
}

output "artifact_bucket_name" {
  value = module.compute.artifact_bucket_name
}

output "instance_ids" {
  value = module.compute.instance_ids
}

output "gpu_classes" {
  value = module.compute.gpu_classes
}

output "worker_pools" {
  value = module.compute.worker_pools
}
