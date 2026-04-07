locals {
  kubeconfig_path = "/etc/rancher/k3s/k3s.yaml"
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = var.artifact_bucket_name
  force_destroy = var.force_destroy_bucket

  tags = merge(var.tags, {
    Name = "${var.name}-artifacts"
  })
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

data "template_file" "controller_user_data" {
  template = file("${path.module}/templates/controller-cloud-init.tftpl")

  vars = {
    cluster_token = var.cluster_token
    region        = var.region
  }
}

data "template_file" "worker_user_data" {
  count    = var.worker_count
  template = file("${path.module}/templates/worker-cloud-init.tftpl")

  vars = {
    cluster_token     = var.cluster_token
    controller_private_ip = aws_instance.controller.private_ip
    region            = var.region
  }
}

resource "aws_instance" "controller" {
  ami                         = var.ami_id
  instance_type               = var.controller_instance_type
  subnet_id                   = var.subnet_ids[0]
  vpc_security_group_ids      = [var.security_group_id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  user_data                   = data.template_file.controller_user_data.rendered

  root_block_device {
    volume_size = var.root_volume_size
    volume_type = "gp3"
  }

  tags = merge(var.tags, {
    Name         = "${var.name}-controller"
    Role         = "controller"
    HourlyRate   = tostring(var.controller_hourly_rate)
    BenchmarkRun = var.benchmark_run_id
  })
}

resource "aws_instance" "workers" {
  count = var.worker_count

  ami                         = var.ami_id
  instance_type               = var.worker_instance_type
  subnet_id                   = var.subnet_ids[count.index % length(var.subnet_ids)]
  vpc_security_group_ids      = [var.security_group_id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  user_data                   = data.template_file.worker_user_data[count.index].rendered

  root_block_device {
    volume_size = var.root_volume_size
    volume_type = "gp3"
  }

  tags = merge(var.tags, {
    Name         = "${var.name}-worker-${count.index + 1}"
    Role         = "worker"
    HourlyRate   = tostring(var.worker_hourly_rate)
    BenchmarkRun = var.benchmark_run_id
  })
}
