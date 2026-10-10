"""Bước 3.1 (ngoại lai) + 3.2 (biến đổi): các transformer học tham số trên tập Train."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .tien_xu_ly import DAC_TRUNG_SO, DAC_TRUNG_NHI_PHAN


class CatNgoaiLai(BaseEstimator, TransformerMixin):
    """Winsorizing: cắt giá trị ngoại lai về phân vị [thap, cao] học từ tập Train.

    Không xóa dòng ngoại lai vì các giá trị cực trị (VD Creatinine > 1000 µmol/L ở
    bệnh nhân suy thận) là có thật về mặt lâm sàng; chỉ giới hạn để chúng không làm
    lệch mô hình (nhất là Logistic Regression / SVM nhạy với thang đo).
    Không dùng IQR vì nhiều cột có IQR = 0 (giá trị lặp lại quá nhiều).
    """

    def __init__(self, thap=0.01, cao=0.99):
        self.thap = thap
        self.cao = cao

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.thap, axis=0)
        self.hi_ = np.nanquantile(X, self.cao, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def tao_bo_tien_xu_ly():
    so = Pipeline([
        ("dien_thieu", SimpleImputer(strategy="median")),  # dữ liệu thiếu -> median (Train)
        ("ngoai_lai", CatNgoaiLai(0.01, 0.99)),             # ngoại lai -> cắt P1-P99
        ("chuan_hoa", StandardScaler()),                    # chuẩn hóa z-score
    ])
    return ColumnTransformer([
        ("so", so, DAC_TRUNG_SO),
        ("nhi_phan", "passthrough", DAC_TRUNG_NHI_PHAN),    # đã mã hóa 0/1
    ])
