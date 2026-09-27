from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ibm_hr_job_satisfaction_tree.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# โมเดลนี้ (DecisionTreeClassifier) ทำนาย JobSatisfaction แบบ Multiclass
# ฟีเจอร์ที่ใช้ (ตามลำดับใน model.feature_names_in_): JobLevel, Age, MonthlyIncome, YearsAtCompany
# ทั้งหมดเป็น "ค่าจริง" ไม่ต้อง normalize (Decision Tree ไม่ไวต่อสเกลของข้อมูล)
# classes_ = [1, 2, 3, 4] -> แปลความหมายเป็น Low / Medium / High / Very High
LABEL_MAP = {
    1: "ต่ำ (Low)",
    2: "ปานกลาง (Medium)",
    3: "สูง (High)",
    4: "สูงมาก (Very High)"
}

# ---------- หน้าเว็บ ----------
st.set_page_config(page_title="Job Satisfaction Predictor", page_icon="😊", layout="centered")
st.title("😊 IBM HR — Job Satisfaction Level Predictor")
st.write("กรอกข้อมูลพนักงานด้านล่าง แล้วกดปุ่มทำนายระดับความพึงพอใจในการทำงาน")

# ---------- รับ input จากผู้ใช้ ----------
col1, col2 = st.columns(2)

with col1:
    job_level = st.selectbox(
        "ระดับตำแหน่งงาน (JobLevel)", options=[1, 2, 3, 4, 5], index=0,
        help="1 = ระดับเริ่มต้น ... 5 = ระดับสูงสุด"
    )
    age = st.slider("อายุ (Age)", min_value=18, max_value=60, value=35)

with col2:
    monthly_income = st.number_input(
        "เงินเดือน (MonthlyIncome)", min_value=1000, max_value=20000, value=5000, step=100
    )
    years_at_company = st.slider("จำนวนปีที่อยู่บริษัทนี้ (YearsAtCompany)", min_value=0, max_value=40, value=5)

# ---------- จัดข้อมูลให้ตรงกับฟีเจอร์ของโมเดล ----------
input_df = pd.DataFrame([{
    "JobLevel": job_level,
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
}])[model.feature_names_in_]  # จัดลำดับคอลัมน์ให้ตรงกับตอนเทรนโมเดล

with st.expander("🔎 ดูข้อมูลที่จะส่งให้โมเดล"):
    st.dataframe(input_df, use_container_width=True)

# ---------- ทำนายผล ----------
st.divider()
if st.button("🔮 ทำนายผล", type="primary"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    st.success(f"ระดับความพึงพอใจในการทำงานที่ทำนายได้: **{LABEL_MAP[prediction]}**")

    st.write("ความน่าจะเป็นของแต่ละระดับ:")
    proba_df = pd.DataFrame({
        "ระดับ": [LABEL_MAP[c] for c in classes],
        "ความน่าจะเป็น (%)": [round(p * 100, 1) for p in proba],
    }).set_index("ระดับ")
    st.bar_chart(proba_df)

st.caption("โมเดล: Decision Tree (max_depth=3) — ฝึกจากข้อมูล IBM HR Analytics Employee Attrition & Performance")

# ---------- Footer ----------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "จัดทำโดย น.ส.กมวรรณ จันทร์ผึ้ง001&nbsp;&nbsp;"
    "น.ส.ชลดา อิศรเสนา ณ อยุธยา009&nbsp;&nbsp;"
    "น.ส.ณัฐพร เผือกผ่อง016"
    "</p>",
    unsafe_allow_html=True,
)
