"""
Bước 3.1 - Làm sạch dữ liệu (các thao tác theo từng dòng, không "học" từ dữ liệu).

Các thao tác cần học thống kê từ dữ liệu (điền median, cắt ngoại lai theo phân vị,
chuẩn hóa) được đặt trong sklearn Pipeline ở train.py và CHỈ fit trên tập Train
để tránh rò rỉ dữ liệu (data leakage) sang Val/Test.
"""
import numpy as np
import pandas as pd

TARGET = "NguyCoTieuDuong"

# Các chỉ số xét nghiệm (đơn vị theo dữ liệu gốc)
COT_SO = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL", "Triglycerid",
          "Creatinine", "Ure", "AcidUric", "AST", "ALT"]

# Giá trị HbA1C = 8.025 xuất hiện ở ~96% bản ghi: đây là giá trị được điền sẵn
# (không phải kết quả đo thật). Ta coi nó là "không đo" và tạo cờ HbA1C_DoThuc.
HBA1C_DIEN_SAN = 8.025

# Cột LDL có 99.7% giá trị = 3.45 -> gần như hằng số, không mang thông tin -> loại bỏ.
COT_LOAI_BO = ["LDL"]

# Đặc trưng đưa vào mô hình
DAC_TRUNG_SO = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "Triglycerid",
                "Creatinine", "Ure", "AcidUric", "AST", "ALT"]
DAC_TRUNG_NHI_PHAN = ["GioiTinh_Nam", "HbA1C_DoThuc"]
DAC_TRUNG = DAC_TRUNG_SO + DAC_TRUNG_NHI_PHAN

# Ngưỡng sinh lý: vượt "max" coi là lỗi nhập liệu (thường do mất dấu thập phân)
NGUONG_SINH_LY = {
    "Tuoi": (0, 120),
    "Glucose": (1, 50),         # mmol/L (40.4 ở BN tăng đường huyết nặng là có thật)
    "HbA1C": (3, 20),           # %
    "Cholesterol": (1, 20),     # mmol/L
    "HDL": (0.1, 5),
    "LDL": (0.1, 15),
    "Triglycerid": (0.1, 80),
    "Creatinine": (10, 2000),   # µmol/L (trẻ nhỏ có thể ~10-20)
    "Ure": (0.5, 80),           # mmol/L
    "AcidUric": (50, 1500),     # µmol/L
    "AST": (1, 5000),           # U/L
    "ALT": (1, 5000),
}

# Chuẩn hóa nhãn về 0/1
MAP_NHAN = {
    "true": 1, "1": 1, "có nguy cơ": 1, "co nguy co": 1, "yes": 1,
    "false": 0, "0": 0, "không có nguy cơ": 0, "khong co nguy co": 0, "no": 0,
}
# Chuẩn hóa giới tính về Nam / Nữ
MAP_GIOI_TINH = {"nam": "Nam", "male": "Nam", "m": "Nam",
                 "nữ": "Nữ", "nu": "Nữ", "female": "Nữ", "f": "Nữ"}


def doc_du_lieu(duong_dan):
    return pd.read_csv(duong_dan, encoding="utf-8-sig", dtype=str)


def sua_lech_thap_phan(x, lo, hi):
    """Giá trị > hi: thử chia 10 liên tiếp (lỗi mất dấu phẩy, VD 750 -> 7.5, 8025 -> 8.025).
    Nếu vẫn không hợp lệ hoặc < lo -> coi là lỗi, trả về NaN."""
    if pd.isna(x):
        return x
    v = x
    while v > hi:
        v /= 10
    return v if lo <= v <= hi else np.nan


def lam_sach(df, nhat_ky=None):
    """Làm sạch theo từng dòng. Trả về DataFrame mới. nhat_ky (list) ghi lại các thay đổi."""
    log = nhat_ky if nhat_ky is not None else []
    df = df.copy()

    # 1) Đồng bộ kiểu dữ liệu số (chuỗi lỗi -> NaN)
    for c in COT_SO:
        if c in df.columns:
            truoc = df[c].notna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if (d := truoc - df[c].notna().sum()):
                log.append(f"{c}: {d} giá trị không phải số -> NaN")

    # 2) Đồng bộ giới tính (Male/Female -> Nam/Nữ)
    if "GioiTinh" in df.columns:
        g = df["GioiTinh"].astype(str).str.strip().str.lower().map(MAP_GIOI_TINH)
        doi = ((df["GioiTinh"].notna()) & (g != df["GioiTinh"])).sum()
        log.append(f"GioiTinh: đồng bộ {doi} giá trị Male/Female -> Nam/Nữ")
        df["GioiTinh"] = g  # giá trị lạ / thiếu -> NaN, xử lý ở bước điền thiếu

    # 3) Đồng bộ nhãn (TRUE/1/'Có nguy cơ' -> 1; FALSE/0/'Không có nguy cơ' -> 0)
    if TARGET in df.columns:
        y = df[TARGET].astype(str).str.strip().str.lower().map(MAP_NHAN)
        log.append(f"{TARGET}: chuẩn hóa nhãn về 0/1; "
                   f"{y.isna().sum()} nhãn không nhận dạng được (bị loại)")
        df[TARGET] = y
        df = df[df[TARGET].notna()].copy()
        df[TARGET] = df[TARGET].astype(int)

    # 4) Dữ liệu lỗi: ngoài ngưỡng sinh lý -> sửa lệch thập phân hoặc NaN
    for c, (lo, hi) in NGUONG_SINH_LY.items():
        if c not in df.columns:
            continue
        cu = df[c].copy()
        df[c] = df[c].apply(lambda v: sua_lech_thap_phan(v, lo, hi))
        thay = cu.notna() & (cu != df[c])
        for i in df.index[thay]:
            log.append(f"{c} dòng {i + 2}: {cu[i]} -> {df.at[i, c]}")

    # 5) HbA1C điền sẵn 8.025 -> không đo thật: tạo cờ, đặt NaN để điền lại sau
    df["HbA1C_DoThuc"] = (df["HbA1C"].notna() & ~np.isclose(df["HbA1C"], HBA1C_DIEN_SAN)).astype(int)
    log.append(f"HbA1C: {int((1 - df['HbA1C_DoThuc']).sum())} bản ghi là giá trị điền sẵn "
               f"{HBA1C_DIEN_SAN}/thiếu -> đánh dấu HbA1C_DoThuc=0")
    df.loc[df["HbA1C_DoThuc"] == 0, "HbA1C"] = np.nan

    # 6) Mã hóa giới tính (one-hot nhị phân). Thiếu -> giá trị phổ biến nhất (Nam)
    so_thieu = df["GioiTinh"].isna().sum()
    df["GioiTinh"] = df["GioiTinh"].fillna("Nam")
    log.append(f"GioiTinh: điền {so_thieu} giá trị thiếu bằng mode = 'Nam'")
    df["GioiTinh_Nam"] = (df["GioiTinh"] == "Nam").astype(int)

    # 7) Loại cột không mang thông tin
    df = df.drop(columns=[c for c in COT_LOAI_BO if c in df.columns])
    return df


def bang_ban_ghi(gia_tri: dict) -> pd.DataFrame:
    """Chuyển thông tin 1 bệnh nhân (từ giao diện) thành DataFrame đã làm sạch."""
    d = {c: [gia_tri.get(c)] for c in COT_SO}
    d["GioiTinh"] = [gia_tri.get("GioiTinh")]
    return lam_sach(pd.DataFrame(d))[DAC_TRUNG]
