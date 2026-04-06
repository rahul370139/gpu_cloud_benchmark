output "controller_public_ip" {
  value = aws_instance.controller.public_ip
}

output "controller_private_ip" {
  value = aws_instance.controller.private_ip
}

output "worker_public_ips" {
  value = aws_instance.workers[*].public_ip
}

output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "instance_ids" {
  value = concat([aws_instance.controller.id], aws_instance.workers[*].id)
}
