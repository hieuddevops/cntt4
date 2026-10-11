"""Bước 7: giao diện nhập thông tin, cảnh báo và giải thích.  Chạy: streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from xu_ly import DAC_TRUNG, lam_sach

st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")
goi = st.cache_resource(joblib.load)(Path(__file__).parent / "outputs" / "mo_hinh.joblib")
pipe = goi["pipeline"]

# (tên hiển thị, đơn vị, mặc định, min, max, ngưỡng bình thường [thấp, cao])
CHI_SO = {
    "Glucose": ("Glucose lúc đói", "mmol/L", 5.0, 1.0, 50.0, (3.9, 5.5)),
    "Cholesterol": ("Cholesterol", "mmol/L", 4.8, 1.0, 20.0, (0, 5.2)),
    "HDL": ("HDL", "mmol/L", 1.4, 0.1, 5.0, (1.0, 99)),
    "Triglycerid": ("Triglycerid", "mmol/L", 1.4, 0.1, 80.0, (0, 1.7)),
    "Creatinine": ("Creatinine", "µmol/L", 80.0, 10.0, 2000.0, (44, 106)),
    "Ure": ("Ure", "mmol/L", 5.0, 0.5, 80.0, (2.5, 7.5)),
    "AcidUric": ("Acid uric", "µmol/L", 330.0, 50.0, 1500.0, (150, 420)),
    "AST": ("AST", "U/L", 25.0, 1.0, 5000.0, (0, 40)),
    "ALT": ("ALT", "U/L", 22.0, 1.0, 5000.0, (0, 40)),
}

st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.caption(f"Mô hình: {goi['ten']} · Công cụ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ.")

with st.form("form"):
    cot = st.columns(3)
    v = {"Tuoi": cot[0].number_input("Tuổi", 1, 120, 50),
         "GioiTinh": cot[0].radio("Giới tính", ["Nam", "Nữ"], horizontal=True)}
    co_hba1c = cot[0].checkbox("Có xét nghiệm HbA1C")
    hba1c = cot[0].number_input("HbA1C (%) – dùng khi đã tích ô trên", 3.0, 20.0, 5.5, 0.1)
    for i, (k, (ten, dv, md, lo, hi, _)) in enumerate(CHI_SO.items()):
        v[k] = cot[1 + i % 2].number_input(f"{ten} ({dv})", lo, hi, md, 0.1)
    gui = st.form_submit_button("🔍 Dự đoán", type="primary")

if gui:
    v["HbA1C"] = hba1c if co_hba1c else None
    X = lam_sach(pd.DataFrame([v]))[DAC_TRUNG]
    p = pipe.predict_proba(X)[0, 1]

    # Mức cảnh báo = max(mô hình, luật chuyên gia ADA)
    muc_ml = 2 if p >= 0.7 else 1 if p >= 0.3 else 0
    g, h = v["Glucose"], v["HbA1C"] or 0
    luat = 2 if g >= 7 or h >= 6.5 else 1 if g >= 5.6 or h >= 5.7 else 0
    muc = max(muc_ml, luat)

    c1, c2 = st.columns([1, 2])
    c1.metric("Xác suất có nguy cơ (mô hình)", f"{p:.1%}")
    if luat > muc_ml:
        c1.caption("ℹ️ Mức cảnh báo được nâng theo tiêu chuẩn ADA.")
    [c2.success, c2.warning, c2.error][muc]([
        "✅ **NGUY CƠ THẤP** – Duy trì lối sống lành mạnh, khám định kỳ 6–12 tháng.",
        "⚠️ **NGUY CƠ TRUNG BÌNH** – Xét nghiệm lại sau 1–3 tháng, giảm đường/tinh bột, vận động ≥150 phút/tuần.",
        "🚨 **NGUY CƠ CAO** – Nên đến cơ sở y tế xét nghiệm khẳng định và được bác sĩ tư vấn.",
    ][muc])

    # Giải thích 1: đối chiếu khoảng tham chiếu
    st.subheader("Giải thích kết quả")
    dong = []
    if g >= 7: dong.append(("Glucose", g, "🔴 ≥ 7.0: đạt ngưỡng chẩn đoán ĐTĐ (ADA)"))
    elif g >= 5.6: dong.append(("Glucose", g, "🟠 5.6–6.9: tiền đái tháo đường"))
    if v["HbA1C"] is not None:
        dong.append(("HbA1C", h, "🔴 ≥ 6.5%: đạt ngưỡng chẩn đoán ĐTĐ" if h >= 6.5 else
                     "🟠 5.7–6.4%: tiền đái tháo đường" if h >= 5.7 else "🟢 Bình thường"))
    for k, (ten, dv, _, _, _, (lo, hi)) in CHI_SO.items():
        if k == "Glucose" and g >= 5.6:
            continue
        nx = "🟠 Cao" if v[k] > hi else "🟠 Thấp" if v[k] < lo else "🟢 Bình thường"
        dong.append((ten, v[k], nx))
    e1, e2 = st.columns(2)
    e1.markdown("**Đối chiếu chỉ số với khoảng tham chiếu**")
    e1.dataframe(pd.DataFrame(dong, columns=["Chỉ số", "Giá trị", "Đánh giá"]), hide_index=True)

    # Giải thích 2: thay từng đặc trưng bằng trung vị Train, xem xác suất đổi bao nhiêu
    dg = {c: (p - pipe.predict_proba(X.assign(**{c: goi["trung_vi"][c]}))[0, 1]) * 100 for c in DAC_TRUNG}
    dg = pd.Series(dg).loc[lambda s: s.abs() > 0.5].sort_values(key=abs, ascending=False).head(6)
    e2.markdown("**Yếu tố ảnh hưởng nhiều nhất đến dự đoán (điểm %)**")
    if dg.empty:
        e2.info("Không có chỉ số nào làm thay đổi đáng kể nguy cơ so với người có chỉ số trung vị.")
    else:
        e2.bar_chart(dg, horizontal=True)
        e2.caption("Dương: làm tăng nguy cơ · Âm: làm giảm nguy cơ (so với người có chỉ số trung vị).")
