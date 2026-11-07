## 1. Tổng hợp kết quả của 3 bài tập:
|                   | **A1: Logistic (NumPy)** | **A2: Softmax (NumPy)** | **A3 Binary (sklearn)** | **A3 Multiclass (sklearn)** |
| ----------------- | ------------------------ | ----------------------- | ----------------------- | --------------------------- |
| **Số lớp**        | 2 (bỏ rating 3)          | 5 (cả 5 rating)         | 2 (bỏ rating 3)         | 5 (cả 5 rating)             |
| **Loss cuối**     | 0.6909                   | 1.5976                  | —                       | —                           |
| **Loss baseline** | ln(2) ≈ 0.6931           | ln(5) ≈ 1.6094          | —                       | —                           |
| **Precision**     | 0.4783                   | 0.1319                  | 0.5000                  | 0.2454                      |
| **Recall**        | 0.4400                   | 0.1483                  | 0.1467                  | 0.2059                      |
| **F1-Score**      | 0.4583                   | 0.1137                  | 0.2268                  | 0.1955                      |
## 2. Phân tích
- Dataset books_new.csv được crawl từ web books_toscape.com có rating được gán ngẫu nhiên, mặc dù đã được feature engineering nhưng kết quả eval lại hoàn toàn cho thấy dữ liệu không có tín hiệu phân biệt rating.
- Cả hai đều gần bằng loss của model đoán ngẫu nhiên đều -> model chỉ học được tỉ lệ lớp (prior), không học được ranh giới phân biệt.
