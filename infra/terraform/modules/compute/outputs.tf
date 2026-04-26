output "controller_public_ip" {
  value = aws_instance.controller.public_ip
}

output "controller_private_ip" {
  value = aws_instance.controller.private_ip
}

output "worker_public_ips" {
  value = [for w in aws_instance.workers : w.public_ip]
}

output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "instance_ids" {
  value = concat(
    [aws_instance.controller.id],
    [for w in aws_instance.workers : w.id],
  )
}

# One entry per GPU class actually provisioned, used by the K8s Job renderer
# to target each class with a matching nodeSelector.
output "gpu_classes" {
  value = distinct([for w in var.worker_pools : w.gpu_class])
}

output "worker_pools" {
  value = var.worker_pools
}
