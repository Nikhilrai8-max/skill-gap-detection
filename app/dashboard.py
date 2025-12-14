import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PATH, PCA_MODEL_PATH, SCALER_PATH
from src.curriculum_optimizer import recommend_clos_for_skill_row
from src.skill_gap import (
    compute_skill_gap,
    student_gap_vs_target,
    interpret_gap_scalar
)

st.set_page_config(layout='wide', page_title='SkillGap PCA Dashboard')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    pcs_path = Path(__file__).parent.parent / 'data' / 'skill_pca_scores.csv'
    pcs = pd.read_csv(pcs_path) if pcs_path.exists() else pd.DataFrame()
    scaler = joblib.load(SCALER_PATH) if Path(SCALER_PATH).exists() else None
    pca_model = joblib.load(PCA_MODEL_PATH) if Path(PCA_MODEL_PATH).exists() else None
    return df, pcs, scaler, pca_model

skills, pcs, scaler, pca_model = load_data()

# Tabs
if pcs.empty:
    tabs = st.tabs(["📊 Skill Gap Overview", "📈 PCA Projection", "🧑‍🎓 Student Input Gap Checker", "📡 Radar Visualization"])
else:
    tabs = st.tabs(["📊 Skill Gap Overview", "📈 PCA Projection", "🧑‍🎓 Student Input Gap Checker", "📡 Radar Visualization"])

tab1, tab2, tab3, tab4 = tabs

with tab1:
    st.title("📊 Skill Gap Overview")
    skills['gap'] = skills['demand_score'] - skills['supply_score']
    selected = st.selectbox("Select a skill", skills['skill'].tolist())
    row = skills[skills['skill'] == selected].iloc[0]
    st.metric('Demand', int(row['demand_score']))
    st.metric('Supply', int(row['supply_score']))
    st.metric('Gap', int(row['gap']))
    st.subheader('CLO Recommendation')
    missing = recommend_clos_for_skill_row(row)
    if missing:
        st.write('Recommended CLO improvements:', missing)
    else:
        st.write('Curriculum already covers this skill well.')
    st.subheader('Top 10 Skill Gaps')
    st.table(skills.sort_values('gap', ascending=False)[['skill','demand_score','supply_score','gap']].head(10))

with tab2:
    st.title('📈 PCA Projection (Skills)')
    if pcs.empty:
        st.write('PCA scores not found. Run main.py to generate them.')
    else:
        merged = pcs.merge(skills[['skill','gap']], on='skill')
        chart = alt.Chart(merged).mark_circle(size=80).encode(x='PC1', y='PC2', tooltip=['skill','gap'], color='gap')
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.title('🧑‍🎓 Student Skill Gap Checker')
    st.write('Enter your scores (0–100) for the following areas:')
    s_supply = st.slider('Overall self-rated skill (0-100)', 0, 100, 50)

    # Core CLO sliders
    clo_vals = {}
    clo_vals['CLO_Programming'] = st.slider('Programming', 0, 100, 50)
    clo_vals['CLO_DSA'] = st.slider('Data Structures & Algorithms', 0, 100, 50)
    clo_vals['CLO_DBMS'] = st.slider('DBMS Concepts', 0, 100, 50)
    clo_vals['CLO_WebDev'] = st.slider('Web Development', 0, 100, 50)
    clo_vals['CLO_Cloud'] = st.slider('Cloud / DevOps', 0, 100, 50)

    # Additional skills (20)
    st.subheader('Additional Skills')
    additional = {}
    additional['Machine Learning'] = st.slider('Machine Learning', 0, 100, 50)
    additional['Deep Learning'] = st.slider('Deep Learning', 0, 100, 50)
    additional['Data Analysis'] = st.slider('Data Analysis', 0, 100, 50)
    additional['Power BI'] = st.slider('Power BI', 0, 100, 50)
    additional['Tableau'] = st.slider('Tableau', 0, 100, 50)
    additional['API Security'] = st.slider('API Security', 0, 100, 50)
    additional['Ethical Hacking'] = st.slider('Ethical Hacking', 0, 100, 50)
    additional['Network Security'] = st.slider('Network Security', 0, 100, 50)
    additional['Linux Administration'] = st.slider('Linux Administration', 0, 100, 50)
    additional['Bash Scripting'] = st.slider('Bash Scripting', 0, 100, 50)
    additional['Android Development'] = st.slider('Android Development', 0, 100, 50)
    additional['UI/UX Design'] = st.slider('UI/UX Design', 0, 100, 50)
    additional['DevOps Fundamentals'] = st.slider('DevOps Fundamentals', 0, 100, 50)
    additional['Cloud Architecture'] = st.slider('Cloud Architecture', 0, 100, 50)
    additional['System Administration'] = st.slider('System Administration', 0, 100, 50)
    additional['Communication'] = st.slider('Communication', 0, 100, 50)
    additional['Leadership'] = st.slider('Leadership', 0, 100, 50)
    additional['Problem Solving'] = st.slider('Problem Solving', 0, 100, 50)
    additional['Critical Thinking'] = st.slider('Critical Thinking', 0, 100, 50)
    additional['Team Collaboration'] = st.slider('Team Collaboration', 0, 100, 50)

    if st.button('Calculate Gap'):
        # Prepare student vector: order must match feature columns used in preprocessing
        feature_order = ['demand_score','supply_score','CLO_Programming','CLO_DSA','CLO_DBMS','CLO_WebDev','CLO_Cloud']
        student_vector = [0, s_supply, clo_vals['CLO_Programming'], clo_vals['CLO_DSA'], clo_vals['CLO_DBMS'], clo_vals['CLO_WebDev'], clo_vals['CLO_Cloud']]
        # scale and pca
        if scaler is None or pca_model is None:
            st.error('PCA model or scaler not found. Run main.py first.')
        else:
            import numpy as np
            student_scaled = scaler.transform([student_vector])
            student_pcs = pca_model.transform(student_scaled)[0]
            industry_target = pcs[['PC1','PC2','PC3']].mean().values
            gap_data = student_gap_vs_target(student_pcs, industry_target)
            st.subheader('PCA Gap Result')
            st.write('Gap Vector:', gap_data['gap_vector'])
            st.write('L2 Gap:', round(gap_data['gap_l2'],3))
            st.write('Mahalanobis Approx:', round(gap_data['gap_mahalanobis_approx'],3))
            st.subheader('Interpretation')
            st.success(interpret_gap_scalar(gap_data['gap_l2']))

with tab4:
    st.title('📡 Radar Chart Visualization')
    st.write('This radar chart compares Industry Demand vs Supply (averages).')
    import matplotlib.pyplot as plt
    labels = ['Demand','Supply']
    demand_avg = skills['demand_score'].mean()
    supply_avg = skills['supply_score'].mean()
    values = [demand_avg, supply_avg]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Industry Demand vs Supply Radar Chart')
    st.pyplot(fig)
    st.subheader('Student vs Industry PCA Radar (example)')
    if pcs.empty:
        st.write('Run main.py to generate PCA scores first.')
    else:
        student_pcs = np.array([50,50,50])
        industry_target = pcs[['PC1','PC2','PC3']].mean().values
        labels = ['PC1','PC2','PC3']
        s_vals = student_pcs.tolist() + [student_pcs[0]]
        t_vals = industry_target.tolist() + [industry_target[0]]
        ang = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
        ang += ang[:1]
        fig2 = plt.figure(figsize=(6,6))
        ax2 = plt.subplot(111, polar=True)
        ax2.plot(ang, s_vals, linewidth=2, label='Student')
        ax2.fill(ang, s_vals, alpha=0.2)
        ax2.plot(ang, t_vals, linewidth=2, label='Industry')
        ax2.fill(ang, t_vals, alpha=0.2)
        ax2.set_thetagrids(np.degrees(ang[:-1]), labels)
        ax2.legend(loc='upper right')
        st.pyplot(fig2)
