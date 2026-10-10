"""
Bước 7 - Giao diện nhập thông tin người bệnh, sinh cảnh báo và giải thích kết quả.

Chạy:  streamlit run app.py      (cần chạy train.py trước để có outputs/mo_hinh_tot_nhat.joblib)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.tien_xu_ly import bang_ban_ghi

GOC = Path(__file__).parent
st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")


@st.cache_resource
def tai_mo_hinh():
    return joblib.load(GOC / "outputs" / "mo_hinh_tot_nhat.joblib")


# Khoảng tham chiếu lâm sàng (người trưởng thành) dùng để giải thích từng chỉ số
def danh_gia_chi_so(v, gioi_tinh):
    nam = gioi_tinh == "Nam"
    kq = []

    def them(ten, gt, don_vi, muc, nhan_xet):
        kq.append({"Chỉ số": ten, "Giá trị": f"{gt:g} {don_vi}", "Mức": muc, "Nhận xét": nhan_xet})

    g = v["Glucose"]
    if g >= 7.0:
        them("Glucose", g, "mmol/L", "🔴 Cao", "≥ 7.0: đạt ngưỡng chẩn đoán đái tháo đường (ADA)")
    elif g >= 5.6:
        them("Glucose", g, "mmol/L", "🟠 Hơi cao", "5.6–6.9: rối loạn đường huyết lúc đói (tiền ĐTĐ)")
    elif g < 3.9:
        them("Glucose", g, "mmol/L", "🟠 Thấp", "< 3.9: hạ đường huyết")
    else:
        them("Glucose", g, "mmol/L", "🟢 Bình thường", "3.9–5.5 mmol/L")

    if v["HbA1C"] is not None:
        h = v["HbA1C"]
        if h >= 6.5:
            them("HbA1C", h, "%", "🔴 Cao", "≥ 6.5%: đạt ngưỡng chẩn đoán đái tháo đường")
        elif h >= 5.7:
            them("HbA1C", h, "%", "🟠 Hơi cao", "5.7–6.4%: tiền đái tháo đường")
        else:
            them("HbA1C", h, "%", "🟢 Bình thường", "< 5.7%")

    def nguong(ten, gt, don_vi, lo, hi, ghi_chu_cao, ghi_chu_thap="Thấp hơn bình thường"):
        if gt > hi:
            them(ten, gt, don_vi, "🟠 Cao", f"> {hi:g}: {ghi_chu_cao}")
        elif gt < lo:
            them(ten, gt, don_vi, "🟠 Thấp", f"< {lo:g}: {ghi_chu_thap}")
        else:
            them(ten, gt, don_vi, "🟢 Bình thường", f"{lo:g}–{hi:g} {don_vi}")

    nguong("Cholesterol", v["Cholesterol"], "mmol/L", 0, 5.2, "tăng cholesterol máu")
    nguong("HDL", v["HDL"], "mmol/L", 1.0 if nam else 1.3, 99, "",
           "HDL thấp – tăng nguy cơ tim mạch/kháng insulin")
    nguong("Triglycerid", v["Triglycerid"], "mmol/L", 0, 1.7, "tăng triglycerid – liên quan hội chứng chuyển hóa")
    nguong("Creatinine", v["Creatinine"], "µmol/L", 62 if nam else 44, 106 if nam else 80,
           "có thể suy giảm chức năng thận")
    nguong("Ure", v["Ure"], "mmol/L", 2.5, 7.5, "có thể suy giảm chức năng thận")
    nguong("Acid uric", v["AcidUric"], "µmol/L", 180 if nam else 150, 420 if nam else 360, "tăng acid uric máu")
    nguong("AST", v["AST"], "U/L", 0, 40, "men gan tăng")
    nguong("ALT", v["ALT"], "U/L", 0, 40, "men gan tăng (gan nhiễm mỡ hay gặp ở người ĐTĐ)")
    return pd.DataFrame(kq)


def dong_gop_cuc_bo(pipe, X, trung_vi):
    """Giải thích cục bộ: thay lần lượt từng đặc trưng bằng giá trị trung vị của tập Train,
    đo xác suất thay đổi bao nhiêu -> mức đóng góp của đặc trưng đó vào dự đoán."""
    p0 = pipe.predict_proba(X)[0, 1]
    rows = []
    for c in X.columns:
        Xc = X.copy()
        Xc[c] = trung_vi[c]
        rows.append({"Đặc trưng": c, "Đóng góp (điểm %)": round((p0 - pipe.predict_proba(Xc)[0, 1]) * 100, 1)})
    return pd.DataFrame(rows).sort_values("Đóng góp (điểm %)", key=abs, ascending=False)


# ------------------------------------------------------------------ Giao diện
st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.caption("Phần mềm hỗ trợ sàng lọc bằng học máy – KHÔNG thay thế chẩn đoán của bác sĩ.")

try:
    goi = tai_mo_hinh()
except FileNotFoundError:
    st.error("Chưa có mô hình. Hãy chạy `python train.py` trước.")
    st.stop()
st.sidebar.success(f"Mô hình đang dùng: **{goi['ten']}**")

with st.form("nhap_lieu"):
    st.subheader("Thông tin người bệnh")
    c1, c2, c3 = st.columns(3)
    with c1:
        ho_ten = st.text_input("Họ tên (không bắt buộc)")
        tuoi = st.number_input("Tuổi", 1, 120, 50)
        gioi_tinh = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
        glucose = st.number_input("Glucose lúc đói (mmol/L)", 1.0, 50.0, 5.6, 0.1)
        co_hba1c = st.checkbox("Có xét nghiệm HbA1C")
        # Trong st.form widget không cập nhật tức thì -> không dùng disabled, chỉ dùng khi đã tích chọn
        hba1c = st.number_input("HbA1C (%) – chỉ dùng khi đã tích ô trên", 3.0, 20.0, 5.5, 0.1)
    with c2:
        chol = st.number_input("Cholesterol (mmol/L)", 1.0, 20.0, 4.8, 0.1)
        hdl = st.number_input("HDL (mmol/L)", 0.1, 5.0, 1.4, 0.1)
        tg = st.number_input("Triglycerid (mmol/L)", 0.1, 80.0, 1.4, 0.1)
        creat = st.number_input("Creatinine (µmol/L)", 10.0, 2000.0, 80.0, 0.1)
    with c3:
        ure = st.number_input("Ure (mmol/L)", 0.5, 80.0, 5.0, 0.1)
        acid = st.number_input("Acid uric (µmol/L)", 50.0, 1500.0, 330.0, 0.1)
        ast = st.number_input("AST (U/L)", 1.0, 5000.0, 25.0, 0.1)
        alt = st.number_input("ALT (U/L)", 1.0, 5000.0, 22.0, 0.1)
    gui = st.form_submit_button("🔍 Dự đoán", type="primary")

if gui:
    gia_tri = {"Tuoi": tuoi, "GioiTinh": gioi_tinh, "Glucose": glucose,
               "HbA1C": hba1c if co_hba1c else None, "Cholesterol": chol, "HDL": hdl,
               "Triglycerid": tg, "Creatinine": creat, "Ure": ure, "AcidUric": acid, "AST": ast, "ALT": alt}
    X = bang_ban_ghi(gia_tri)
    pipe = goi["pipeline"]
    p = pipe.predict_proba(X)[0, 1]

    st.divider()
    st.subheader(f"Kết quả{' – ' + ho_ten if ho_ten else ''}")
    k1, k2 = st.columns([1, 2])
    k1.metric("Xác suất có nguy cơ tiểu đường (mô hình)", f"{p:.1%}")
    k1.progress(float(p))

    # Kết hợp mô hình học máy + luật chuyên gia (tiêu chuẩn ADA): lấy mức cảnh báo cao hơn
    muc_ml = 2 if p >= 0.7 else 1 if p >= 0.3 else 0
    hb = gia_tri["HbA1C"]
    if glucose >= 7.0 or (hb is not None and hb >= 6.5):
        muc_luat, ly_do_luat = 2, "chỉ số đạt ngưỡng chẩn đoán đái tháo đường theo ADA"
    elif glucose >= 5.6 or (hb is not None and hb >= 5.7):
        muc_luat, ly_do_luat = 1, "chỉ số ở vùng tiền đái tháo đường theo ADA"
    else:
        muc_luat, ly_do_luat = 0, ""
    muc = max(muc_ml, muc_luat)
    if muc_luat > muc_ml:
        k1.caption(f"ℹ️ Mức cảnh báo được nâng theo luật chuyên gia: {ly_do_luat}.")

    if muc == 2:
        k2.error("### ⚠️ CẢNH BÁO: NGUY CƠ CAO\nCác chỉ số cho thấy khả năng cao mắc đái tháo đường. "
                 "Nên đến cơ sở y tế để làm xét nghiệm khẳng định (Glucose đói lặp lại, HbA1C, "
                 "nghiệm pháp dung nạp glucose) và được bác sĩ tư vấn điều trị.")
    elif muc == 1:
        k2.warning("### ⚠️ NGUY CƠ TRUNG BÌNH\nCó dấu hiệu cần theo dõi. Nên xét nghiệm lại trong 1–3 tháng, "
                   "điều chỉnh chế độ ăn (giảm đường, tinh bột tinh chế), tăng vận động ≥150 phút/tuần.")
    else:
        k2.success("### ✅ NGUY CƠ THẤP\nChưa thấy dấu hiệu nguy cơ rõ rệt. Duy trì lối sống lành mạnh "
                   "và kiểm tra sức khỏe định kỳ 6–12 tháng.")

    st.subheader("Giải thích kết quả")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**1. Đối chiếu từng chỉ số với khoảng tham chiếu lâm sàng**")
        bang = danh_gia_chi_so(gia_tri, gioi_tinh)
        st.dataframe(bang, hide_index=True, width="stretch")
    with e2:
        st.markdown("**2. Yếu tố ảnh hưởng nhiều nhất đến dự đoán của mô hình**")
        dg = dong_gop_cuc_bo(pipe, X, goi["trung_vi_train"])
        dg_top = dg[dg["Đóng góp (điểm %)"] != 0].head(6)
        if dg_top.empty:
            st.info("Không có đặc trưng nào làm thay đổi đáng kể xác suất so với một người 'điển hình'.")
        else:
            st.bar_chart(dg_top.set_index("Đặc trưng"), horizontal=True)
            st.caption("Giá trị dương: đặc trưng này làm TĂNG nguy cơ so với người có chỉ số trung vị; "
                       "âm: làm GIẢM nguy cơ.")

    # Tóm tắt bằng lời
    ly_do = [f"{r['Chỉ số']} = {r['Giá trị']} ({r['Nhận xét']})"
             for _, r in bang.iterrows() if r["Mức"].startswith("🔴")
             or (r["Mức"].startswith("🟠") and r["Chỉ số"] in ("Glucose", "HbA1C"))]
    st.markdown("**3. Tóm tắt**")
    if ly_do:
        st.write("Mô hình đánh giá nguy cơ chủ yếu dựa trên: " + "; ".join(ly_do) + ".")
    else:
        st.write("Glucose (và HbA1C nếu có) nằm trong giới hạn bình thường – đây là các yếu tố quyết định "
                 "nhất theo mô hình (Glucose chiếm ~83% mức độ quan trọng).")
    khac = [r["Chỉ số"] for _, r in bang.iterrows()
            if r["Mức"].startswith("🟠") and r["Chỉ số"] not in ("Glucose", "HbA1C")]
    if khac:
        st.write("Các chỉ số bất thường khác cần lưu ý: " + ", ".join(khac) + ".")
