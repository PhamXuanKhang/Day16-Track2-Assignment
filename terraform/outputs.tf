output "bastion_public_ip" {
  description = "Public IP of the Bastion host — SSH entry point"
  value       = aws_instance.bastion.public_ip
}

output "cpu_node_private_ip" {
  description = "Private IP of the LightGBM CPU node"
  value       = aws_instance.gpu_node.private_ip
}

output "ssh_bastion_command" {
  description = "Command to SSH into the Bastion host"
  value       = "ssh -i lab-key ubuntu@${aws_instance.bastion.public_ip}"
}

output "ssh_cpu_node_command" {
  description = "Command to SSH into the CPU node (via Bastion)"
  value       = "ssh -i lab-key -J ubuntu@${aws_instance.bastion.public_ip} ubuntu@${aws_instance.gpu_node.private_ip}"
}
