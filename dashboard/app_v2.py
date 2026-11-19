import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthetic_data import generate_synthetic_data, SUBJECTS

st.set_page_config(page_title="Student Intelligence | XAI", page_icon="🎓", layout="wide")

# --- Visual styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
[data-testid="stAppViewContainer"] {background:#0b1220;color:#e8eef7;}
[data-testid="stSidebar"] {background:#101a2b;border-right:1px solid #24334a;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#17263a,#1b3048);padding:16px;border:1px solid #2a3d57;border-radius:14px;}
div[data-testid="stMetricValue"] {color:#f1f6fc;}
div[data-testid="stMetricLabel"] {color:#9fb3c8;}
h1,h2,h3 {color:#f1f6fc !important;}
.stTabs [data-baseweb="tab"] {color:#a9bdd3;}
.small-muted {color:#91a7bf;font-size:0.88rem;}
.hero {background:linear-gradient(120deg,#14243a,#153b50 55%,#16535b);padding:25px 28px;border-radius:18px;border:1px solid #2d5267;margin-bottom:18px;}
.hero h1 {margin:0;color:#f6fbff !important;font-size:1.9rem;}
.hero p {color:#b9d0e2;margin:8px 0 0;}
.tag {display:inline-block;padding:4px 9px;border-radius:99px;background:#4c3a17;color:#ffd47a;font-size:0.75rem;font-weight:700;letter-spacing:.04em;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Generating simulated student records…")
def get_synthetic(n, seed):
    return generate_synthetic_data(n_students=n, seed=seed)

with st.sidebar:
    st.markdown("### 🎓 Student Intelligence")
    st.markdown('<span class="tag">SYNTHETIC DATA MODE</span>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Student Explorer", "Subject Analytics", "Dataset Studio"])
    st.markdown("---")
    n_students = st.slider("Synthetic student profiles", 500, 10000, 2500, 500)
    seed = st.number_input("Generator seed", min_value=0, max_value=999999, value=42, step=1)
    st.caption("Changes to student count or seed regenerate the synthetic dataset.")

df = get_synthetic(n_students, seed)
student_count = df["student_id"].nunique()
pass_rate = df["pass"].mean() * 100
avg_mark = df["final_mark"].mean()
high_risk = (df["risk_category"] == "High").mean() * 100

st.markdown("""
<div class="hero">
  <h1>Student Intelligence Dashboard</h1>
  <p>Academic performance exploration · subject insights · explainable AI project workspace</p>
  <div style="margin-top:12px"><span class="tag">SIMULATED DATA — NOT REAL STUDENT RECORDS</span></div>
</div>
""", unsafe_allow_html=True)

# Sidebar dataset is generated on-demand; all pages use the same data state.
if page == "Overview":
    st.subheader("Performance snapshot")
    a,b,c,d = st.columns(4)
    a.metric("Synthetic profiles", f"{student_count:,}")
    b.metric("Student–subject records", f"{len(df):,}")
    c.metric("Average mark", f"{avg_mark:.1f}/100")
    d.metric("Pass rate", f"{pass_rate:.1f}%")
    st.caption(f"Seed: {seed} · {len(SUBJECTS)} subjects · synthetic records only")
    left,right = st.columns([1.15,1])
    with left:
        subject_summary = df.groupby("subject",as_index=False).agg(mean_mark=("final_mark","mean"),pass_rate=("pass","mean"))
        fig=px.bar(subject_summary.sort_values("mean_mark"),x="mean_mark",y="subject",orientation="h",
                    color="mean_mark",color_continuous_scale=["#e87969","#f3c969","#50d2b2"],
                    labels={"mean_mark":"Average mark","subject":""},title="Average marks by subject")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         coloraxis_showscale=False,margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        risk_counts=df["risk_category"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig=px.pie(values=risk_counts.values,names=risk_counts.index,hole=.62,
                    color=risk_counts.index,color_discrete_map={"Low":"#46c8a5","Medium":"#f0bd61","High":"#e87969"},
                    title="Performance risk distribution")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Pass rate by subject")
    sr=df.groupby("subject",as_index=False)["pass"].mean()
    sr["pass_pct"]=sr["pass"]*100
    fig=px.bar(sr,x="subject",y="pass_pct",color="pass_pct",color_continuous_scale=["#e87969","#f0bd61","#46c8a5"])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                     yaxis_title="Pass rate (%)",xaxis_title="",coloraxis_showscale=False,margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.info("These KPIs and charts describe the generated synthetic dataset. They are not model evaluation results.")

elif page == "Student Explorer":
    st.subheader("Explore synthetic student-subject records")
    filters1,filters2,filters3=st.columns(3)
    with filters1: chosen_subject=st.selectbox("Subject",["All"]+SUBJECTS)
    with filters2: chosen_risk=st.selectbox("Risk category",["All","Low","Medium","High"])
    with filters3: min_att=st.slider("Minimum attendance (%)",0,100,0)
    filtered=df.copy()
    if chosen_subject!="All": filtered=filtered[filtered.subject==chosen_subject]
    if chosen_risk!="All": filtered=filtered[filtered.risk_category==chosen_risk]
    filtered=filtered[filtered.attendance_pct>=min_att]
    st.caption(f"{len(filtered):,} records match the filters.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered records (CSV)",filtered.to_csv(index=False).encode("utf-8"),
                      file_name="synthetic_student_records_filtered.csv",mime="text/csv")

elif page == "Subject Analytics":
    st.subheader("Subject-level comparisons")
    chosen=st.multiselect("Select subjects",SUBJECTS,default=SUBJECTS)
    sub=df[df.subject.isin(chosen)]
    if sub.empty:
        st.warning("Select at least one subject to view analytics.")
    else:
        summary=sub.groupby("subject",as_index=False).agg(
            average_mark=("final_mark","mean"),median_mark=("final_mark","median"),
            pass_rate=("pass","mean"),average_attendance=("attendance_pct","mean"),
            avg_study_hours=("study_hours_week","mean")
        )
        summary["pass_rate_pct"]=summary["pass_rate"]*100
        summary["pass_rate"]=summary["pass_rate_pct"].round(1)
        st.dataframe(summary.drop(columns="pass_rate_pct").round(2),use_container_width=True,hide_index=True)
        fig=px.box(sub,x="subject",y="final_mark",color="subject",title="Mark distribution by subject")
        fig.update_layout(template="plotly_dark",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         yaxis_title="Final mark (0–100)",xaxis_title="",margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig,use_container_width=True)
        fig=px.scatter(summary,x="avg_study_hours",y="average_mark",size="average_attendance",color="subject",
                       hover_name="subject",labels={"avg_study_hours":"Average study hours/week","average_mark":"Average mark"},
                       title="Study time vs subject average (aggregated synthetic groups)")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Subject difficulty offsets and student-level random variation are built into the simulator. This is illustrative, not evidence about real courses.")

elif page == "Dataset Studio":
    st.subheader("Synthetic dataset studio")
    st.write("Generate repeatable fictional student records for UI development, test cases, and controlled experiments.")
    st.warning("Synthetic data is algorithmically generated. It must not be mixed into real-data evaluation without a clearly designed experiment and explicit labeling.")
    col1,col2=st.columns(2)
    with col1:
        st.metric("Profile count",f"{student_count:,}")
        st.metric("Subject count",len(SUBJECTS))
    with col2:
        st.metric("Rows generated",f"{len(df):,}")
        st.metric("Seed",seed)
    st.write("**Subjects included**")
    st.write(" · ".join(SUBJECTS))
    st.write("**Dataset preview**")
    st.dataframe(df.head(30),use_container_width=True,hide_index=True)
    st.download_button("Download full synthetic dataset (CSV)",df.to_csv(index=False).encode("utf-8"),
                      file_name=f"synthetic_students_{student_count}_seed{seed}.csv",mime="text/csv")
    st.markdown("**Generator design**")
    st.markdown("- Student profile attributes are shared across subjects.\n- Subject-specific offsets simulate different score difficulty levels.\n- Marks include noise so students with similar attributes are not identical.\n- PASS threshold is 50/100; grade bands are descriptive synthetic labels.")

st.markdown("---")
st.markdown('<p class="small-muted">XAI Project V2 · synthetic dataset workspace · academic decision support only</p>',unsafe_allow_html=True)
