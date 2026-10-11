# BÀI KIỂM TRA MÔN TRÍ TUỆ NHÂN TẠO

**Đề tài:** Xây dựng phần mềm AI (Machine Learning) dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe

**Sinh viên:** Đào Đức Hiếu

**Mã sinh viên:** 26ACN021

**Dùng thử trực tuyến:** https://hieuddevops-cntt4.streamlit.app/

## PHẦN A. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH

### 1. Cấu trúc thư mục bài nộp

```
README.md                      Tài liệu này: giải thích và hướng dẫn thực hiện
DaoDucHieu_CNTT4/
├── data/CSDL_TieuDuong.csv    Dữ liệu gốc (Bước 2)
├── xu_ly.py                   Tiền xử lý: làm sạch, mã hóa, điền thiếu, ngoại lai, chuẩn hóa (Bước 3)
├── train.py                   Chia dữ liệu, huấn luyện và đánh giá 3 mô hình (Bước 3.3 – 6)
├── app.py                     Giao diện nhập thông tin, cảnh báo và giải thích (Bước 7)
├── outputs/                   Mô hình đã huấn luyện, bảng kết quả, biểu đồ
└── requirements.txt           Danh sách thư viện cần cài
```

### 2. Cài đặt (Python 3.10 trở lên)

```
cd DaoDucHieu_CNTT4
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Huấn luyện và đánh giá

```
python train.py
```

Chương trình in kết quả ra màn hình và lưu vào thư mục DaoDucHieu_CNTT4/outputs/ gồm mo_hinh.joblib, ket_qua_test.csv và 3 biểu đồ .png. Thời gian chạy khoảng 10 giây.

### 4. Chạy giao diện

```
streamlit run app.py
```

Trình duyệt tự mở địa chỉ http://localhost:8501. Nhập các chỉ số xét nghiệm, bấm "Dự đoán" để xem xác suất nguy cơ, mức cảnh báo và phần giải thích. Nếu chỉ số HbA1C có được xét nghiệm, tích ô "Có xét nghiệm HbA1C" rồi nhập giá trị.

## PHẦN B. GIẢI THÍCH CÁC BƯỚC THỰC HIỆN

### Bước 1. Xác định bài toán

- **Loại bài toán:** phân loại nhị phân (học có giám sát).
- **Đầu vào:** 13 thông số gồm tuổi, giới tính, Glucose, HbA1C, Cholesterol, HDL, LDL, Triglycerid, Creatinine, Ure, Acid uric, AST, ALT.
- **Đầu ra:** NguyCoTieuDuong, với 1 là có nguy cơ và 0 là không có nguy cơ, kèm xác suất dự đoán.
- **Mục tiêu:** hỗ trợ sàng lọc sớm. Trong y tế, bỏ sót người bệnh (False Negative) nguy hiểm hơn báo nhầm (False Positive). Vì vậy ngoài Accuracy, bài làm ưu tiên **Recall** và **F1-score**.

### Bước 2. Thu thập dữ liệu

Tệp CSDL_TieuDuong.csv có **4.707 bản ghi × 14 cột**, không có bản ghi trùng lặp. Tỷ lệ nhãn: khoảng 76,7% không nguy cơ và 23,3% có nguy cơ, tức dữ liệu **mất cân bằng**.

### Bước 3. Tiền xử lý dữ liệu

#### 3.1. Làm sạch dữ liệu

**a) Kiểm tra và xử lý dữ liệu thiếu.** Số ô trống ban đầu: Tuổi 1, Giới tính 4, Glucose 1, HbA1C 2, Cholesterol 2, LDL 2, Triglycerid 3, Creatinine 1, Acid uric 5, AST 2, ALT 2.

- Biến số: điền bằng **trung vị (median)** của tập Train. Median ít bị ảnh hưởng bởi ngoại lai hơn trung bình.
- Giới tính: điền bằng **mode** (giá trị phổ biến nhất) là "Nam".
- Không xóa dòng thiếu, vì số lượng ít và xóa đi sẽ mất thông tin của các cột còn lại.

**b) Đồng bộ dữ liệu.**

- Giới tính có lẫn tiếng Anh: Male/Female (11 giá trị) được chuyển thành Nam/Nữ.
- Nhãn có 3 cách ghi: TRUE/FALSE, 1/0, "Có nguy cơ"/"Không có nguy cơ". Tất cả được chuẩn hóa về 1/0.
- Các cột số được ép về kiểu số thực. Chuỗi không hợp lệ được chuyển thành giá trị thiếu.

**c) Kiểm tra và xử lý dữ liệu lỗi.** Mỗi chỉ số được so với ngưỡng sinh lý khả dĩ. Giá trị vượt ngưỡng trên phần lớn do **mất dấu thập phân** khi nhập liệu, nên được sửa bằng cách chia 10 cho tới khi hợp lý:

| Chỉ số | Giá trị lỗi | Sau sửa | Lý do |
|---|---|---|---|
| Glucose | 750; 59; 61; 92 | 7.5; 5.9; 6.1; 9.2 | Glucose máu không thể vượt 50 mmol/L. Giá trị sau sửa khớp với nhãn của bản ghi |
| HbA1C | 8025 | 8.025 | HbA1C tối đa khoảng 20% |
| Cholesterol | 55 | 5.5 | Cholesterol tối đa khoảng 20 mmol/L |
| Creatinine | 4.2; 5.7 | (thiếu) rồi điền median | Thấp bất thường, không thể là creatinine máu |

Giá trị Glucose = 40.4 mmol/L được **giữ nguyên**. Đây là tăng đường huyết rất nặng nhưng có thật về lâm sàng, và bản ghi này có nhãn "có nguy cơ".

**Phát hiện quan trọng về dữ liệu điền sẵn:**

- HbA1C = 8.025 xuất hiện ở **96%** bản ghi. Đây là giá trị bị điền sẵn (giá trị trung bình), không phải kết quả đo. Bài làm coi các giá trị này là "không đo", tạo thêm đặc trưng **HbA1C_DoThuc** (1 là có đo, 0 là không đo) và điền lại bằng median của các giá trị đo thật. Nếu giữ 8.025, mô hình sẽ hiểu sai rằng 96% bệnh nhân có HbA1C vượt ngưỡng tiểu đường 6.5%.
- LDL = 3.45 ở **99,7%** bản ghi, gần như hằng số nên không mang thông tin. Cột này bị **loại bỏ**.
- Các cột khác cũng có giá trị lặp lại nhiều do điền sẵn (Cholesterol 5.5, Triglycerid 2.1, Acid uric 426.2), nhưng vẫn còn đủ biến thiên nên được giữ lại.

**d) Kiểm tra và xử lý dữ liệu ngoại lai.** Dữ liệu có nhiều ngoại lai ở Creatinine (tới 1.511 µmol/L), AST/ALT (tới 534/580 U/L), Triglycerid… Đây là các giá trị **có thật** ở bệnh nhân suy thận, viêm gan, rối loạn mỡ máu nặng, nên **không xóa**. Thay vào đó, bài làm dùng kỹ thuật **Winsorizing**: cắt giá trị về khoảng phân vị 1% đến 99%, với ngưỡng học từ tập Train. Phương pháp IQR không phù hợp vì nhiều cột có IQR = 0 do giá trị lặp lại; dùng IQR sẽ biến mọi giá trị khác giá trị lặp thành "ngoại lai".

#### 3.2. Biến đổi dữ liệu

- **Mã hóa:** Giới tính được chuyển thành biến nhị phân GioiTinh_Nam (1 là Nam, 0 là Nữ). Nhãn được chuyển thành 1/0.
- **Chuẩn hóa:** dùng StandardScaler (z-score: x' = (x − μ)/σ), đưa các biến về cùng thang đo. Bước này cần thiết cho Logistic Regression và SVM vì các chỉ số có thang rất khác nhau (Glucose khoảng 5, Acid uric khoảng 400).
- **Chống rò rỉ dữ liệu (data leakage):** các bước phải "học" từ dữ liệu (median, phân vị, μ và σ) được đóng gói trong sklearn **Pipeline**, chỉ `fit` trên tập Train và sau đó áp dụng cho Val/Test. Nhờ vậy, kết quả đánh giá phản ánh đúng khả năng dự đoán trên dữ liệu mới.

#### 3.3. Phân chia dữ liệu

Chia **phân tầng theo nhãn** (stratified) để tỷ lệ nguy cơ ở các tập như nhau (khoảng 23,3%):

| Tập | Số bản ghi | Tỷ lệ | Vai trò |
|---|---|---|---|
| Train | 3.294 | 70% | Huấn luyện mô hình |
| Validation | 706 | 15% | Chọn siêu tham số, chọn mô hình tốt nhất |
| Test | 707 | 15% | Đánh giá cuối cùng, không dùng khi chọn mô hình |

### Bước 4. Chọn mô hình

| Mô hình | Lý do chọn |
|---|---|
| Logistic Regression | Mô hình tuyến tính cơ sở, dễ giải thích |
| Random Forest | Tập hợp nhiều cây quyết định, mô hình hóa được ngưỡng và tương tác phi tuyến, ít cần chuẩn hóa |
| SVM (kernel RBF) | Hiệu quả với ranh giới phi tuyến, dữ liệu vừa phải |

Do dữ liệu mất cân bằng, cả 3 mô hình đều dùng `class_weight="balanced"` để tăng trọng số cho lớp "có nguy cơ".

### Bước 5. Huấn luyện mô hình

Cả 3 mô hình được huấn luyện trên tập Train. Siêu tham số được chọn bằng cách thử nhiều giá trị và giữ bộ cho **F1 cao nhất trên tập Validation**:

| Mô hình | Siêu tham số |
|---|---|
| Logistic Regression | C = 0.1 |
| Random Forest | n_estimators = 200 |
| SVM (RBF) | C = 0.5, gamma = 0.05 (xác suất được hiệu chỉnh bằng CalibratedClassifierCV) |

### Bước 6. Đánh giá mô hình

**Các chỉ số đánh giá:**

- **Accuracy** = (TP+TN)/Tổng: tỷ lệ dự đoán đúng. Dễ bị "ảo" khi dữ liệu mất cân bằng.
- **Precision** = TP/(TP+FP): trong số ca được báo "có nguy cơ", bao nhiêu ca đúng.
- **Recall** = TP/(TP+FN): trong số người thực sự có nguy cơ, mô hình phát hiện được bao nhiêu. Đây là **chỉ số quan trọng nhất trong sàng lọc y tế**.
- **F1-score:** trung bình điều hòa của Precision và Recall, cân bằng giữa hai chỉ số.
- **ROC-AUC:** khả năng phân biệt hai lớp ở mọi ngưỡng.

**Kết quả trên tập Test (707 bản ghi):**

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9830 | 0.9422 | 0.9879 | 0.9645 | 0.9935 |
| **Random Forest** | **0.9972** | **0.9880** | **1.0000** | **0.9940** | **1.0000** |
| SVM (RBF) | 0.9731 | 0.9679 | 0.9152 | 0.9408 | 0.9952 |

**Mô hình được chọn: Random Forest.** Mô hình này có F1 cao nhất trên tập Validation (1.0), và việc chọn được thực hiện trước khi nhìn tập Test. Trên tập Test, ma trận nhầm lẫn là TN = 540, FP = 2, **FN = 0**, TP = 165: mô hình không bỏ sót người có nguy cơ nào, chỉ báo nhầm 2 trường hợp.

![So sánh mô hình](DaoDucHieu_CNTT4/outputs/so_sanh_mo_hinh.png)

![Ma trận nhầm lẫn](DaoDucHieu_CNTT4/outputs/ma_tran_nham_lan.png)

**Nhận xét:**

- SVM có Recall thấp nhất (0.915), tức bỏ sót khoảng 8,5% người có nguy cơ. Trong sàng lọc y tế, đây là nhược điểm lớn.
- Random Forest vượt trội vì nhãn được xác định bằng **ngưỡng**, mà cây quyết định học ngưỡng rất tự nhiên.
- **Lý do độ chính xác gần 100%:** kiểm tra cho thấy nhãn trong dữ liệu khớp **100%** với tiêu chuẩn chẩn đoán của Hiệp hội Đái tháo đường Hoa Kỳ (ADA): *Glucose đói ≥ 7.0 mmol/L hoặc HbA1C ≥ 6.5%*. Như vậy mô hình đã "học lại" đúng tiêu chuẩn y khoa từ dữ liệu. Glucose chiếm khoảng 83% mức độ quan trọng (hình dưới).

![Độ quan trọng đặc trưng](DaoDucHieu_CNTT4/outputs/do_quan_trong.png)

- **Hạn chế:** mô hình chưa dự báo được nguy cơ *trước khi* đường huyết vượt ngưỡng. Muốn làm được điều đó cần dữ liệu theo dõi theo thời gian và các yếu tố như BMI, huyết áp, tiền sử gia đình. Ngoài ra, chỉ khoảng 170 bản ghi có HbA1C đo thật, nên mô hình học chỉ số này còn yếu.

### Bước 7. Giao diện, cảnh báo và giải thích kết quả

Giao diện web được xây dựng bằng **Streamlit** (app.py):

1. **Nhập liệu:** tuổi, giới tính và các chỉ số xét nghiệm. HbA1C không bắt buộc.
2. **Dự đoán:** hiển thị xác suất có nguy cơ do mô hình Random Forest tính ra.
3. **Sinh cảnh báo theo 3 mức:**
   - 🔴 **Nguy cơ cao** (xác suất ≥ 70%): khuyên đến cơ sở y tế xét nghiệm khẳng định.
   - 🟠 **Nguy cơ trung bình** (30–70%): khuyên theo dõi, xét nghiệm lại sau 1–3 tháng, điều chỉnh lối sống.
   - 🟢 **Nguy cơ thấp** (< 30%): duy trì lối sống lành mạnh, khám định kỳ.

   Bài làm kết hợp **mô hình học máy với luật chuyên gia** (tiêu chuẩn ADA). Mức cảnh báo cuối cùng là mức cao hơn giữa hai nguồn. Ví dụ, người có Glucose 6.5 mmol/L (tiền tiểu đường) được mô hình cho nguy cơ thấp, vì nhãn chỉ đánh dấu ca đã đạt ngưỡng tiểu đường; luật chuyên gia nâng cảnh báo lên "trung bình" để người dùng biết cần theo dõi.
4. **Giải thích kết quả:**
   - Đối chiếu **từng chỉ số với khoảng tham chiếu lâm sàng** (bình thường / cao / thấp), đặc biệt nêu rõ Glucose, HbA1C ở vùng tiền tiểu đường hay đã đạt ngưỡng tiểu đường.
   - **Giải thích cục bộ:** lần lượt thay từng đặc trưng của bệnh nhân bằng giá trị trung vị của tập Train và đo xem xác suất thay đổi bao nhiêu. Kết quả cho biết đặc trưng nào làm tăng hoặc giảm nguy cơ của chính bệnh nhân đó.

**Ví dụ kiểm thử:**

| Ca | Glucose | HbA1C | Xác suất (mô hình) | Cảnh báo |
|---|---|---|---|---|
| Bình thường | 5.0 | – | 0.5% | Nguy cơ thấp |
| Tiền tiểu đường | 6.5 | – | 1.0% | Nguy cơ trung bình (nâng theo luật ADA) |
| Glucose cao | 9.5 | – | 98.5% | Nguy cơ cao |
| HbA1C cao | 5.5 | 7.2 | 66.0% | Nguy cơ cao (nâng theo luật ADA) |

*Lưu ý: phần mềm chỉ có giá trị hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ.*
