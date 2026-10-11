"""Bước 2 -> 6: đọc dữ liệu, tiền xử lý, chia tập, huấn luyện và đánh giá mô hình.
Chạy: python train.py   (kết quả lưu trong outputs/)
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from xu_ly import DAC_TRUNG, NHAN, bo_tien_xu_ly, lam_sach

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Bước 2-3.1: đọc và làm sạch
raw = pd.read_csv(Path(__file__).parent / "data" / "CSDL_TieuDuong.csv", encoding="utf-8-sig", dtype=str)
df = lam_sach(raw)
print(f"Dữ liệu: {len(raw)} bản ghi | thiếu ban đầu: {int(raw.isna().sum().sum())} ô | "
      f"tỷ lệ có nguy cơ: {df[NHAN].mean():.1%}")

# Bước 3.3: chia 70% Train / 15% Val / 15% Test, phân tầng theo nhãn
X, y = df[DAC_TRUNG], df[NHAN]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# Bước 4-5: 3 mô hình, class_weight='balanced' vì dữ liệu mất cân bằng (~23% có nguy cơ)
MO_HINH = {
    "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": CalibratedClassifierCV(SVC(C=0.5, gamma=0.05, class_weight="balanced"), ensemble=False),
}


def danh_gia(m, X_, y_):
    p = m.predict_proba(X_)[:, 1]
    d = (p >= 0.5).astype(int)
    return {"Accuracy": accuracy_score(y_, d), "Precision": precision_score(y_, d),
            "Recall": recall_score(y_, d), "F1": f1_score(y_, d), "ROC-AUC": roc_auc_score(y_, p)}


pipes, val, test = {}, {}, {}
for ten, mh in MO_HINH.items():
    pipes[ten] = make_pipeline(bo_tien_xu_ly(), mh).fit(X_train, y_train)
    val[ten], test[ten] = danh_gia(pipes[ten], X_val, y_val), danh_gia(pipes[ten], X_test, y_test)

# Bước 6: chọn mô hình theo F1 trên Val, báo cáo kết quả trên Test
bang = pd.DataFrame(test).T.round(4)
print("\nVal:\n", pd.DataFrame(val).T.round(4), "\n\nTest:\n", bang)
tot = max(val, key=lambda t: val[t]["F1"])
print(f"\n=> Chọn: {tot}")
bang.to_csv(OUT / "ket_qua_test.csv", encoding="utf-8-sig")
joblib.dump({"ten": tot, "pipeline": pipes[tot], "trung_vi": X_train.median().to_dict()},
            OUT / "mo_hinh.joblib")

# Biểu đồ
bang.plot.bar(rot=0, ylim=(0.8, 1.0), figsize=(9, 4), title="So sánh mô hình trên tập Test")
plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.tight_layout(); plt.savefig(OUT / "so_sanh_mo_hinh.png", dpi=120); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (ten, p) in zip(axes, pipes.items()):
    ConfusionMatrixDisplay.from_estimator(p, X_test, y_test, ax=ax, colorbar=False,
                                          display_labels=["Không", "Có"], cmap="Blues")
    ax.set_title(ten)
plt.tight_layout(); plt.savefig(OUT / "ma_tran_nham_lan.png", dpi=120); plt.close()

rf = pipes["Random Forest"][-1]
pd.Series(rf.feature_importances_, index=DAC_TRUNG).sort_values().plot.barh(
    figsize=(6, 4.5), title="Độ quan trọng đặc trưng (Random Forest)")
plt.tight_layout(); plt.savefig(OUT / "do_quan_trong.png", dpi=120); plt.close()
