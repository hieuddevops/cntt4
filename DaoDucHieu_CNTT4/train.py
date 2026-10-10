"""
Dự đoán nguy cơ tiểu đường - Huấn luyện & đánh giá mô hình (Bước 2 -> 6).

Chạy:  python train.py
Kết quả lưu trong thư mục outputs/ (biểu đồ, bảng chỉ số, nhật ký làm sạch, mô hình).
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.bien_doi import tao_bo_tien_xu_ly
from src.tien_xu_ly import DAC_TRUNG, DAC_TRUNG_SO, TARGET, doc_du_lieu, lam_sach

SEED = 42
GOC = Path(__file__).parent
OUT = GOC / "outputs"
OUT.mkdir(exist_ok=True)


def tieu_de(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


# ---------------------------------------------------------------- Bước 2
tieu_de("BƯỚC 2. THU THẬP DỮ LIỆU")
raw = doc_du_lieu(GOC / "data" / "CSDL_TieuDuong.csv")
print(f"Kích thước: {raw.shape[0]} bản ghi x {raw.shape[1]} cột")
print("Phân bố nhãn gốc:\n", raw[TARGET].value_counts().to_string())

# ---------------------------------------------------------------- Bước 3.1
tieu_de("BƯỚC 3.1. LÀM SẠCH DỮ LIỆU")
thieu_goc = raw.replace("", np.nan).isna().sum()
print("Số giá trị thiếu theo cột (dữ liệu gốc):\n", thieu_goc[thieu_goc > 0].to_string())
print("Số bản ghi trùng lặp:", raw.duplicated().sum())

nhat_ky = []
df = lam_sach(raw, nhat_ky)
for dong in nhat_ky:
    print(" -", dong)
(OUT / "nhat_ky_lam_sach.txt").write_text("\n".join(nhat_ky), encoding="utf-8")
df.to_csv(OUT / "du_lieu_sau_lam_sach.csv", index=False, encoding="utf-8-sig")
print(f"Sau làm sạch: {len(df)} bản ghi; nhãn 1 = {df[TARGET].mean():.1%}")
# Kiểm tra tính nhất quán của nhãn với tiêu chuẩn chẩn đoán ADA
ada = (df["Glucose"] >= 7.0) | ((df["HbA1C_DoThuc"] == 1) & (df["HbA1C"] >= 6.5))
print(f"Nhãn khớp tiêu chuẩn ADA (Glucose đói >= 7.0 mmol/L hoặc HbA1C >= 6.5%): "
      f"{(ada.astype(int) == df[TARGET]).mean():.2%}")
print("Giá trị thiếu còn lại (sẽ điền median của tập Train trong Pipeline):")
print(df[DAC_TRUNG].isna().sum()[lambda s: s > 0].to_string())

# Biểu đồ: phân bố nhãn + boxplot ngoại lai
fig, ax = plt.subplots(figsize=(4.5, 3.5))
df[TARGET].map({0: "Không nguy cơ", 1: "Có nguy cơ"}).value_counts().plot.bar(
    ax=ax, color=["#4C78A8", "#E45756"], rot=0)
ax.set_title("Phân bố nhãn sau làm sạch"); ax.set_ylabel("Số bản ghi")
fig.tight_layout(); fig.savefig(OUT / "01_phan_bo_nhan.png", dpi=120); plt.close(fig)

fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for ax, c in zip(axes.ravel(), DAC_TRUNG_SO):
    ax.boxplot(df[c].dropna())
    ax.set_title(c, fontsize=10)
axes.ravel()[-1].axis("off")
fig.suptitle("Boxplot các chỉ số (trước khi cắt ngoại lai P1–P99)")
fig.tight_layout(); fig.savefig(OUT / "02_boxplot_ngoai_lai.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- Bước 3.3
tieu_de("BƯỚC 3.3. PHÂN CHIA DỮ LIỆU 70 / 15 / 15 (phân tầng theo nhãn)")
X, y = df[DAC_TRUNG], df[TARGET]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
for ten, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"{ten:5s}: {len(yy):5d} bản ghi ({len(yy) / len(y):.0%}), tỷ lệ nguy cơ = {yy.mean():.1%}")

# ---------------------------------------------------------------- Bước 4 + 5
tieu_de("BƯỚC 4-5. CHỌN & HUẤN LUYỆN MÔ HÌNH (chọn siêu tham số trên tập Val)")
# Dữ liệu mất cân bằng (~23% có nguy cơ) -> class_weight='balanced'
UNG_VIEN = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                            {"C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                      {"n_estimators": [200, 400], "max_depth": [None, 8, 12], "min_samples_leaf": [1, 3]}),
    # SVC không có xác suất sẵn -> hiệu chỉnh xác suất bằng CalibratedClassifierCV
    "SVM (RBF)": (CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                                         ensemble=False),
                  {"estimator__C": [0.5, 1, 5, 10], "estimator__gamma": ["scale", 0.05]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 9, 15], "weights": ["uniform", "distance"]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
}


def chi_so(model, X_, y_):
    p = model.predict_proba(X_)[:, 1]
    d = (p >= 0.5).astype(int)
    return {"Accuracy": accuracy_score(y_, d), "Precision": precision_score(y_, d, zero_division=0),
            "Recall": recall_score(y_, d), "F1": f1_score(y_, d), "ROC-AUC": roc_auc_score(y_, p)}


mo_hinh, kq_val, sieu_tham_so = {}, {}, {}
for ten, (est, grid) in UNG_VIEN.items():
    tot, tot_f1 = None, -1
    for params in ParameterGrid(grid):
        pipe = Pipeline([("tien_xu_ly", tao_bo_tien_xu_ly()), ("mo_hinh", clone(est).set_params(**params))])
        pipe.fit(X_train, y_train)
        f1 = f1_score(y_val, pipe.predict(X_val))
        if f1 > tot_f1:
            tot, tot_f1, tot_p = pipe, f1, params
    mo_hinh[ten], sieu_tham_so[ten] = tot, tot_p
    kq_val[ten] = chi_so(tot, X_val, y_val)
    print(f"{ten:20s} | tham số tốt nhất: {tot_p} | F1(val) = {tot_f1:.4f}")

bang_val = pd.DataFrame(kq_val).T.round(4)
print("\nKết quả trên tập VALIDATION:\n", bang_val.to_string())

# ---------------------------------------------------------------- Bước 6
tieu_de("BƯỚC 6. ĐÁNH GIÁ TRÊN TẬP TEST (dữ liệu mô hình chưa từng thấy)")
bang_test = pd.DataFrame({t: chi_so(m, X_test, y_test) for t, m in mo_hinh.items()}).T.round(4)
print(bang_test.to_string())
bang_val.to_csv(OUT / "ket_qua_validation.csv", encoding="utf-8-sig")
bang_test.to_csv(OUT / "ket_qua_test.csv", encoding="utf-8-sig")

# Chọn mô hình tốt nhất theo F1 trên Val (Recall làm tiêu chí phụ) - không nhìn Test khi chọn
ten_tot = max(kq_val, key=lambda t: (kq_val[t]["F1"], kq_val[t]["Recall"]))
best = mo_hinh[ten_tot]
print(f"\n=> Mô hình được chọn: {ten_tot}")
d_test = best.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, d_test).ravel()
print(f"Ma trận nhầm lẫn (Test): TN={tn} FP={fp} FN={fn} TP={tp}")

# Biểu đồ so sánh chỉ số
fig, ax = plt.subplots(figsize=(10, 4.5))
bang_test[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot.bar(ax=ax, rot=0, width=0.8)
ax.set_ylim(0.5, 1.0); ax.set_title("So sánh các mô hình trên tập Test")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "03_so_sanh_mo_hinh.png", dpi=120); plt.close(fig)

# Ma trận nhầm lẫn
fig, axes = plt.subplots(1, len(mo_hinh), figsize=(4 * len(mo_hinh), 3.6))
for ax, (t, m) in zip(axes, mo_hinh.items()):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, ax=ax, colorbar=False,
                                          display_labels=["Không", "Có"], cmap="Blues")
    ax.set_title(t, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "04_ma_tran_nham_lan.png", dpi=120); plt.close(fig)

# ROC
fig, ax = plt.subplots(figsize=(5.5, 5))
for t, m in mo_hinh.items():
    fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, label=f"{t} (AUC={bang_test.loc[t, 'ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("Đường cong ROC - tập Test"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "05_roc.png", dpi=120); plt.close(fig)

# Mức độ quan trọng của đặc trưng (Random Forest)
rf = mo_hinh["Random Forest"].named_steps["mo_hinh"]
quan_trong = pd.Series(rf.feature_importances_, index=DAC_TRUNG).sort_values()
fig, ax = plt.subplots(figsize=(6, 4.5))
quan_trong.plot.barh(ax=ax, color="#4C78A8"); ax.set_title("Mức độ quan trọng của đặc trưng (Random Forest)")
fig.tight_layout(); fig.savefig(OUT / "06_do_quan_trong.png", dpi=120); plt.close(fig)
print("\nĐộ quan trọng đặc trưng (RF):\n", quan_trong.sort_values(ascending=False).round(4).to_string())

# Lưu mô hình + thông tin cho giao diện
joblib.dump({"ten": ten_tot, "pipeline": best, "dac_trung": DAC_TRUNG,
             "trung_vi_train": X_train.median().to_dict()}, OUT / "mo_hinh_tot_nhat.joblib")
json.dump({"mo_hinh_chon": ten_tot, "sieu_tham_so": {k: {kk: str(vv) for kk, vv in v.items()}
                                                     for k, v in sieu_tham_so.items()},
           "test": bang_test.to_dict(orient="index"),
           "ma_tran_nham_lan_test": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}},
          open(OUT / "tom_tat.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nĐã lưu mô hình: {OUT / 'mo_hinh_tot_nhat.joblib'}")
