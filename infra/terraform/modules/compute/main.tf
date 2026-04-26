locals {
  kubeconfig_path = "/etc/rancher/k3s/k3s.yaml"

  # Flatten worker_pools into a single list of workers. Each entry carries
  # enough metadata to render cloud-init and tag the EC2 instance.
  workers = flatten([
    for pool in var.worker_pools : [
      for i in range(pool.count) : {
        key           = "${pool.gpu_class}-${i + 1}"
        gpu_class     = pool.gpu_class
        instance_type = pool.instance_type
        hourly_rate   = pool.hourly_rate
        pool_index    = i
      }
    ]
  ])
  workers_by_key = { for w in local.workers : w.key => w }
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

data "aws_iam_policy_document" "node_artifacts" {
  statement {
    effect = "Allow"
    actions = [
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.artifacts.arn,
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = [
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.name}-node-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = merge(var.tags, {
    Name = "${var.name}-node-role"
  })
}

resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "node" {
  name = "${var.name}-node-profile"
  role = aws_iam_role.node.name
}

resource "aws_iam_role_policy" "node_artifacts" {
  name   = "${var.name}-node-artifacts"
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.node_artifacts.json
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

resource "aws_instance" "controller" {
  ami                         = var.controller_ami_id
  instance_type               = var.controller_instance_type
  subnet_id                   = var.subnet_ids[0]
  vpc_security_group_ids      = [var.security_group_id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  user_data = templatefile("${path.module}/templates/controller-cloud-init.tftpl", {
    cluster_token = var.cluster_token
    region        = var.region
  })
  iam_instance_profile = aws_iam_instance_profile.node.name

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
  for_each = local.workers_by_key

  ami                         = var.worker_ami_id
  instance_type               = each.value.instance_type
  subnet_id                   = var.subnet_ids[each.value.pool_index % length(var.subnet_ids)]
  vpc_security_group_ids      = [var.security_group_id]
  key_name                    = var.key_name
  associate_public_ip_address = true
  user_data = templatefile("${path.module}/templates/worker-cloud-init.tftpl", {
    cluster_token         = var.cluster_token
    controller_private_ip = aws_instance.controller.private_ip
    region                = var.region
    gpu_class             = each.value.gpu_class
  })
  iam_instance_profile = aws_iam_instance_profile.node.name

  root_block_device {
    volume_size = var.root_volume_size
    volume_type = "gp3"
  }

  tags = merge(var.tags, {
    Name         = "${var.name}-worker-${each.key}"
    Role         = "worker"
    GpuClass     = each.value.gpu_class
    HourlyRate   = tostring(each.value.hourly_rate)
    BenchmarkRun = var.benchmark_run_id
  })
}
