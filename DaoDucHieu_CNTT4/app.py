"""Bước 7: giao diện nhập thông tin, cảnh báo và giải thích.  Chạy: streamlit run app.py"""
import math
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from xu_ly import DAC_TRUNG, lam_sach

OUT = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")
goi = st.cache_resource(joblib.load)(OUT / "mo_hinh.joblib")
pipe = goi["pipeline"]

# key: (tên, đơn vị, mặc định, min, max, khoảng bình thường, thang hiển thị)
NHOM = {
    "🩸 Đường huyết": {"Glucose": ("Glucose lúc đói", "mmol/L", 5.0, 1.0, 50.0, (3.9, 5.5), (2, 14))},
    "🫀 Mỡ máu": {"Cholesterol": ("Cholesterol", "mmol/L", 4.8, 1.0, 20.0, (2.5, 5.2), (1, 10)),
                 "HDL": ("HDL", "mmol/L", 1.4, 0.1, 5.0, (1.0, 2.5), (0, 3)),
                 "Triglycerid": ("Triglycerid", "mmol/L", 1.4, 0.1, 80.0, (0.4, 1.7), (0, 6))},
    "🫘 Thận": {"Creatinine": ("Creatinine", "µmol/L", 80.0, 10.0, 2000.0, (44, 106), (0, 250)),
               "Ure": ("Ure", "mmol/L", 5.0, 0.5, 80.0, (2.5, 7.5), (0, 20)),
               "AcidUric": ("Acid uric", "µmol/L", 330.0, 50.0, 1500.0, (150, 420), (0, 800))},
    "🧪 Gan": {"AST": ("AST", "U/L", 25.0, 1.0, 5000.0, (5, 40), (0, 150)),
              "ALT": ("ALT", "U/L", 22.0, 1.0, 5000.0, (5, 40), (0, 150))},
}
CHI_SO = {k: v for nhom in NHOM.values() for k, v in nhom.items()}
TEN = {**{k: v[0] for k, v in CHI_SO.items()}, "Tuoi": "Tuổi", "GioiTinh_Nam": "Giới tính",
       "HbA1C_DoThuc": "Có đo HbA1C"}
MAC_DINH = {"Tuoi": 50, "GioiTinh": "Nam", "co_hba1c": False, "HbA1C": 5.5,
            **{k: v[2] for k, v in CHI_SO.items()}}
CA_MAU = {
    "🟢 Bình thường": {},
    "🟠 Tiền ĐTĐ": {"Tuoi": 52, "Glucose": 6.4, "Cholesterol": 5.8, "Triglycerid": 2.3},
    "🔴 Nguy cơ cao": {"Tuoi": 61, "Glucose": 11.2, "co_hba1c": True, "HbA1C": 8.4,
                      "Triglycerid": 3.5, "HDL": 0.9, "ALT": 55.0},
}
for k, v in MAC_DINH.items():
    st.session_state.setdefault(k, v)


def dien_ca_mau():
    if ca := st.session_state.ca_mau:
        st.session_state.update({**MAC_DINH, **CA_MAU[ca]})


def dong_ho(p):
    """Đồng hồ nửa vòng tròn: xanh < 30% <= cam < 70% <= đỏ."""
    def diem(f, r=80):
        t = math.pi * (1 - f)
        return 100 + r * math.cos(t), 100 - r * math.sin(t)

    cung = ""
    for a, b, mau in [(0, .3, "#2ea043"), (.3, .7, "#f0a020"), (.7, 1, "#e5484d")]:
        (x1, y1), (x2, y2) = diem(a), diem(b)
        cung += f'<path d="M{x1:.1f} {y1:.1f} A80 80 0 0 1 {x2:.1f} {y2:.1f}" stroke="{mau}" stroke-width="18" fill="none"/>'
    x, y = diem(p, 62)
    return f"""<svg viewBox="0 0 200 125" style="width:100%;max-width:300px;display:block;margin:auto">
      {cung}<line x1="100" y1="100" x2="{x:.1f}" y2="{y:.1f}" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="7" fill="currentColor"/>
      <text x="100" y="122" text-anchor="middle" font-size="18" font-weight="700" fill="currentColor">{p:.1%}</text></svg>"""


def thanh_tham_chieu(ten, gt, dv, lo, hi, smin, smax):
    """Thanh ngang: vùng xanh = bình thường, chấm = giá trị của bệnh nhân."""
    pct = lambda x: max(0, min(100, (x - smin) / (smax - smin) * 100))
    dtd = {"Glucose lúc đói": 7.0, "HbA1C": 6.5}.get(ten)  # ngưỡng chẩn đoán ĐTĐ (ADA)
    if dtd and hi < gt < dtd:
        mau, nhan = "#f0a020", "Tiền ĐTĐ"
    else:
        mau, nhan = ("#e5484d", "Cao") if gt > hi else ("#f0a020", "Thấp") if gt < lo else ("#2ea043", "Bình thường")
    return f"""<div style="margin:6px 0 14px">
      <div style="display:flex;justify-content:space-between;font-size:14px">
        <span>{ten}</span><span><b>{gt:g}</b> {dv} · <span style="color:{mau}">{nhan}</span></span></div>
      <div style="position:relative;height:8px;border-radius:4px;background:rgba(128,128,128,.2);margin-top:4px">
        <div style="position:absolute;left:{pct(lo)}%;width:{pct(hi) - pct(lo)}%;height:100%;background:rgba(46,160,67,.45);border-radius:4px"></div>
        <div style="position:absolute;left:calc({pct(gt)}% - 7px);top:-3px;width:14px;height:14px;border-radius:50%;background:{mau};border:2px solid white"></div>
      </div><div style="font-size:11px;opacity:.6;margin-top:2px">Bình thường: {lo:g}–{hi:g}</div></div>"""


# ------------------------------------------------------------- Thanh bên: nhập liệu
with st.sidebar:
    st.header("📝 Thông tin người bệnh")
    st.pills("Điền nhanh ca mẫu", list(CA_MAU), key="ca_mau", on_change=dien_ca_mau)
    c1, c2 = st.columns(2)
    c1.number_input("Tuổi", 1, 120, key="Tuoi")
    c2.radio("Giới tính", ["Nam", "Nữ"], key="GioiTinh", horizontal=True)
    for nhom, ds in NHOM.items():
        st.subheader(nhom)
        for k, (ten, dv, _, lo, hi, _, _) in ds.items():
            st.number_input(f"{ten} ({dv})", lo, hi, step=0.1, key=k)
        if nhom.endswith("Đường huyết"):
            st.checkbox("Có xét nghiệm HbA1C", key="co_hba1c")
            st.number_input("HbA1C (%)", 3.0, 20.0, step=0.1, key="HbA1C", disabled=not st.session_state.co_hba1c)

v = {k: st.session_state[k] for k in ["Tuoi", "GioiTinh", *CHI_SO]}
v["HbA1C"] = st.session_state.HbA1C if st.session_state.co_hba1c else None

# ------------------------------------------------------------- Phần chính: kết quả
st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.caption(f"Mô hình {goi['ten']} · Công cụ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ.")
tab_kq, tab_mh = st.tabs(["Kết quả dự đoán", "Về mô hình"])

with tab_kq:
    X = lam_sach(pd.DataFrame([v]))[DAC_TRUNG]
    p = pipe.predict_proba(X)[0, 1]
    # Mức cảnh báo = max(mô hình học máy, luật chuyên gia theo tiêu chuẩn ADA)
    g, h = v["Glucose"], v["HbA1C"] or 0
    muc_ml = 2 if p >= 0.7 else 1 if p >= 0.3 else 0
    luat = 2 if g >= 7 or h >= 6.5 else 1 if g >= 5.6 or h >= 5.7 else 0
    muc = max(muc_ml, luat)

    c1, c2 = st.columns([2, 3], vertical_alignment="center")
    with c1:
        st.markdown(dong_ho(p), unsafe_allow_html=True)
        st.caption("<center>Xác suất có nguy cơ theo mô hình</center>", unsafe_allow_html=True)
    with c2:
        [st.success, st.warning, st.error][muc]([
            "### ✅ Nguy cơ thấp\nChưa thấy dấu hiệu nguy cơ rõ rệt. Duy trì lối sống lành mạnh, khám định kỳ 6–12 tháng.",
            "### ⚠️ Nguy cơ trung bình\nCần theo dõi: xét nghiệm lại sau 1–3 tháng, giảm đường và tinh bột, vận động ≥150 phút/tuần.",
            "### 🚨 Nguy cơ cao\nNên đến cơ sở y tế làm xét nghiệm khẳng định (Glucose đói, HbA1C) và được bác sĩ tư vấn.",
        ][muc])
        if g >= 7 or h >= 6.5:
            st.markdown("**Lý do:** chỉ số đạt ngưỡng chẩn đoán đái tháo đường theo ADA "
                        "(Glucose đói ≥ 7.0 mmol/L hoặc HbA1C ≥ 6.5%).")
        elif g >= 5.6 or h >= 5.7:
            st.markdown("**Lý do:** chỉ số ở vùng tiền đái tháo đường theo ADA "
                        "(Glucose 5.6–6.9 mmol/L hoặc HbA1C 5.7–6.4%).")
        if luat > muc_ml:
            st.caption("ℹ️ Mức cảnh báo được nâng theo luật chuyên gia (tiêu chuẩn ADA).")

    st.divider()
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.subheader("Đối chiếu với khoảng tham chiếu")
        html = "".join(thanh_tham_chieu(t, v[k], dv, lo, hi, *thang)
                       for k, (t, dv, _, _, _, (lo, hi), thang) in CHI_SO.items())
        if v["HbA1C"] is not None:
            html = thanh_tham_chieu("HbA1C", h, "%", 4.0, 5.6, 3, 12) + html
        st.html(html)
    with e2:
        st.subheader("Yếu tố ảnh hưởng đến dự đoán")
        # Thay từng đặc trưng bằng trung vị tập Train, xem xác suất thay đổi bao nhiêu điểm %
        dg = pd.DataFrame([{"Chỉ số": TEN.get(c, c), "Điểm %": round(
            (p - pipe.predict_proba(X.assign(**{c: goi["trung_vi"][c]}))[0, 1]) * 100, 1)} for c in DAC_TRUNG])
        dg = dg[dg["Điểm %"].abs() > 0.5]
        if dg.empty:
            st.info("Không có chỉ số nào làm thay đổi đáng kể nguy cơ so với người có chỉ số trung vị.")
        else:
            st.altair_chart(alt.Chart(dg).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Điểm %:Q", title="Thay đổi xác suất (điểm %)"),
                y=alt.Y("Chỉ số:N", sort="-x", title=None),
                color=alt.condition("datum['Điểm %'] > 0", alt.value("#e5484d"), alt.value("#2ea043")),
                tooltip=["Chỉ số", "Điểm %"]).properties(height=40 * len(dg) + 40))
            st.caption("🔴 Làm tăng nguy cơ · 🟢 Làm giảm nguy cơ (so với người có chỉ số trung vị).")

with tab_mh:
    st.subheader("Kết quả đánh giá trên tập Test (15% dữ liệu)")
    st.dataframe(pd.read_csv(OUT / "ket_qua_test.csv", index_col=0, encoding="utf-8-sig")
                 .style.highlight_max(color="rgba(46,160,67,.35)").format("{:.4f}"))
    st.markdown("Dữ liệu 4.707 bản ghi, chia 70% Train / 15% Val / 15% Test. Mô hình được chọn theo F1 "
                "trên tập Val. Chi tiết tiền xử lý và đánh giá xem README trên GitHub.")
    c1, c2 = st.columns(2)
    c1.image(str(OUT / "so_sanh_mo_hinh.png"))
    c2.image(str(OUT / "do_quan_trong.png"))
    st.image(str(OUT / "ma_tran_nham_lan.png"))
