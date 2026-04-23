# Lab 16 – Cloud AI Environment Setup (CPU / LightGBM Track)

**Instance:** AWS `m7i-flex.large` (2 vCPU, 8 GB RAM) — CPU-only  
**Dataset:** Credit Card Fraud Detection (Kaggle, 284,807 rows)  
**Model:** LightGBM GBDT binary classifier

---

## Lý do chọn CPU thay vì GPU

Tài khoản AWS mới kích hoạt mặc định có hạn mức GPU (`Running On-Demand G and VT instances`) là **0 vCPU**. Yêu cầu tăng hạn mức đã được gửi nhưng chưa được AWS phê duyệt trong thời gian làm lab, nên chuyển sang **CPU fallback plan** theo hướng dẫn Phần 7 của README_aws.md. LightGBM được chọn vì đây là thuật toán gradient boosting tối ưu cho CPU, hỗ trợ đa luồng tốt và phù hợp với bài toán phát hiện gian lận trên dữ liệu dạng bảng.

---

## Kết quả Benchmark

| Metric | Giá trị |
|--------|---------|
| Data load time | 1.0272 s |
| Training time | 27.1139 s |
| Best iteration | 500 (full run — early stopping bị tắt do không có validation set riêng) |
| AUC-ROC | 0.8639 |
| Accuracy | 0.9980 |
| F1-Score | 0.5594 |
| Precision | 0.4479 |
| Recall | 0.7449 |
| Inference latency (1 row) | 0.3334 ms |
| Inference throughput (1000 rows) | 39.38 ms (~25,394 rows/s) |

---

## So sánh CPU vs GPU (phân tích)

**Training (27 giây trên CPU):** LightGBM sử dụng histogram-based splitting và chạy song song trên tất cả CPU core, nên 27 giây để hoàn thành 500 vòng lặp GBDT trên 227,845 mẫu là hoàn toàn hợp lý. GPU thường giúp tăng tốc training khoảng 3–10× với dataset lớn hơn, nhưng với dữ liệu dạng bảng 30 features như bộ này, overhead của GPU transfer có thể làm lợi thế giảm đáng kể.

**Inference (0.33 ms/row, ~25,000 rows/s):** Tốc độ inference trên CPU hoàn toàn đáp ứng yêu cầu production cho bài toán phát hiện gian lận thời gian thực — ngưỡng thông thường là dưới 100 ms/transaction. GPU mang lại lợi thế inference chủ yếu khi batch size lớn (>10,000 rows) và model là deep learning, không phải GBDT.

**Chất lượng model:** AUC-ROC 0.86 là kết quả chấp nhận được nhưng chưa tối ưu. F1-Score thấp (0.56) phản ánh sự mất cân bằng cực lớn của dataset (chỉ 0.17% là gian lận). Có thể cải thiện bằng cách thêm `scale_pos_weight`, dùng validation set riêng để early stopping hoạt động đúng, và tuning threshold phân loại.

**Kết luận:** Với bài toán ML truyền thống trên tabular data, CPU instance như `m7i-flex.large` là lựa chọn **tiết kiệm chi phí** hơn (~$0.10/hr so với ~$0.53/hr của `g4dn.xlarge`) mà không đánh đổi đáng kể về hiệu năng. GPU thực sự cần thiết khi serving các LLM hoặc model deep learning có hàng tỷ tham số.

---

## Kiến trúc hạ tầng (Terraform)

```
Internet
   │
   ▼
[ALB] (Public Subnet, port 80)
   │
   ▼ (port 8000)
[m7i-flex.large] (Private Subnet)
   - Amazon Linux 2023 (Deep Learning AMI)
   - LightGBM + scikit-learn + pandas
   │
[Bastion t3.micro] (Public Subnet, SSH jump host)
   │
[NAT Gateway] → Internet (outbound only for private subnet)
```

**Chi phí ước tính (1 giờ):**

| Tài nguyên | Giá |
|---|---|
| m7i-flex.large | ~$0.101/hr |
| t3.micro (Bastion) | $0.010/hr |
| NAT Gateway | $0.045/hr |
| ALB | $0.008/hr |
| **Tổng** | **~$0.164/hr** |

---

## Hướng dẫn chạy lại

### 1. Chuẩn bị

```bash
# Tạo SSH key
ssh-keygen -t ed25519 -f terraform/lab-key -N ""

# Set AWS credentials
export AWS_ACCESS_KEY_ID=<your_key>
export AWS_SECRET_ACCESS_KEY=<your_secret>
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Deploy

```bash
cd terraform
terraform init
terraform apply
# Lưu lại bastion_public_ip và cpu_node_private_ip từ output
```

### 3. SSH vào CPU node và chạy benchmark

```bash
# Copy benchmark script
scp -i terraform/lab-key -o "ProxyJump ubuntu@<bastion_ip>" \
  terraform/benchmark.py ubuntu@<cpu_node_ip>:~/

# SSH vào node
ssh -i terraform/lab-key -J ubuntu@<bastion_ip> ubuntu@<cpu_node_ip>

# Cấu hình Kaggle
mkdir -p ~/.kaggle
echo '{"username":"<user>","key":"<key>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset và chạy
pip3 install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
python3 ~/benchmark.py
```

### 4. Teardown (bắt buộc)

```bash
cd terraform
terraform destroy
```

---

## File quan trọng

| File | Mô tả |
|------|-------|
| [terraform/main.tf](terraform/main.tf) | Toàn bộ hạ tầng AWS (VPC, EC2, ALB, IAM) |
| [terraform/user_data.sh](terraform/user_data.sh) | Script khởi động EC2 — cài Python + LightGBM |
| [terraform/benchmark.py](terraform/benchmark.py) | Script benchmark LightGBM, xuất JSON |
| [benchmark_result.json](benchmark_result.json) | Kết quả benchmark thực tế |
| [Screenshot_terminal.png](Screenshot_terminal.png) | Screenshot terminal khi chạy benchmark |
