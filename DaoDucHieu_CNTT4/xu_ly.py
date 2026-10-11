"""Bước 3 - Tiền xử lý dữ liệu (dùng chung cho train.py và app.py)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NHAN = "NguyCoTieuDuong"
COT_SO = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "Triglycerid",
          "Creatinine", "Ure", "AcidUric", "AST", "ALT"]
DAC_TRUNG = COT_SO + ["GioiTinh_Nam", "HbA1C_DoThuc"]

# Ngưỡng sinh lý (min, max): vượt max thường do mất dấu thập phân (750 -> 7.5)
NGUONG = {"Tuoi": (0, 120), "Glucose": (1, 50), "HbA1C": (3, 20), "Cholesterol": (1, 20),
          "HDL": (0.1, 5), "Triglycerid": (0.1, 80), "Creatinine": (10, 2000),
          "Ure": (0.5, 80), "AcidUric": (50, 1500), "AST": (1, 5000), "ALT": (1, 5000)}


def sua_loi(v, lo, hi):
    """Chia 10 tới khi <= max; vẫn không hợp lệ thì coi là thiếu."""
    while v > hi:
        v /= 10
    return v if lo <= v <= hi else np.nan


def lam_sach(df):
    """3.1 Làm sạch + mã hóa (các thao tác theo từng dòng)."""
    df = df.copy()
    # Đồng bộ nhãn và giới tính
    if NHAN in df:
        df[NHAN] = df[NHAN].astype(str).str.strip().str.lower().map(
            {"true": 1, "1": 1, "có nguy cơ": 1, "false": 0, "0": 0, "không có nguy cơ": 0})
        df = df.dropna(subset=[NHAN]).astype({NHAN: int})
    gioi = df["GioiTinh"].astype(str).str.strip().str.lower()
    df["GioiTinh_Nam"] = (~gioi.isin(["nữ", "female"])).astype(int)  # thiếu -> mode (Nam)
    # Ép kiểu số + sửa dữ liệu lỗi
    for c, (lo, hi) in NGUONG.items():
        df[c] = pd.to_numeric(df[c], errors="coerce").apply(lambda v: sua_loi(v, lo, hi))
    # HbA1C = 8.025 ở ~96% bản ghi là giá trị điền sẵn, không phải đo thật -> gắn cờ
    df["HbA1C_DoThuc"] = (df["HbA1C"].notna() & ~np.isclose(df["HbA1C"], 8.025)).astype(int)
    df.loc[df["HbA1C_DoThuc"] == 0, "HbA1C"] = np.nan
    return df  # cột LDL (99.7% = 3.45, gần như hằng số) không được dùng


class CatNgoaiLai(BaseEstimator, TransformerMixin):
    """Cắt ngoại lai về phân vị 1%-99% học từ tập Train (không xóa dòng)."""
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanquantile(X, [0.01, 0.99], axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def bo_tien_xu_ly():
    """3.1 điền thiếu + ngoại lai, 3.2 chuẩn hóa. Chỉ fit trên Train để tránh rò rỉ dữ liệu."""
    so = make_pipeline(SimpleImputer(strategy="median"), CatNgoaiLai(), StandardScaler())
    return ColumnTransformer([("so", so, COT_SO)], remainder="passthrough")
