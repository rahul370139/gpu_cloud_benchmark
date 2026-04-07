variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "admin_cidrs" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
