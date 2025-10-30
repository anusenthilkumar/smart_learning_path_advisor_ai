import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

# ---------------------------------------------------------
# Subject dictionary (edit/extend to your syllabus)
# ---------------------------------------------------------
SUBJECT_MAP = {
    "CSE": {
        "Sem 1": ["Engineering Mathematics I", "Physics", "Chemistry", "Programming Fundamentals", "English", "Basic Electrical"],
        "Sem 2": ["Engineering Mathematics II", "Data Structures", "OOP", "Digital Logic", "Environmental Studies"],
        "Sem 3": ["Maths III", "DBMS", "Operating Systems", "Computer Organization", "Data Structures Lab", "OOP Lab"],
        "Sem 4": ["Algorithms", "Computer Networks", "Software Engineering", "Discrete Mathematics", "DBMS Lab", "OS Lab"],
        "Sem 5": ["AI", "Web Technologies", "Compiler Design", "Cloud Computing", "CN Lab", "SE Mini Project"],
        "Sem 6": ["Machine Learning", "Distributed Systems", "Information Security", "Elective I", "ML Lab", "Project Phase I"],
    },
    "ECE": {
        "Sem 1": ["Engineering Mathematics I", "Physics", "Chemistry", "Basic Electronics", "English"],
        "Sem 2": ["Engineering Mathematics II", "Network Theory", "Digital Circuits", "Programming", "Signals & Systems"],
        "Sem 3": ["Electronic Devices", "Analog Circuits", "Maths III", "Control Systems", "Digital Lab", "Devices Lab"],
        "Sem 5": ["Communication Systems", "Microprocessors", "VLSI", "Antenna Theory", "Embedded Lab"],
    },
    "AI&DS": {
        "Sem 3": ["Linear Algebra", "Probability & Stats", "Data Structures", "DBMS", "Python for DS"],
        "Sem 5": ["Machine Learning", "Deep Learning", "Data Mining", "Big Data", "Cloud Computing"],
        "Sem 6": ["NLP", "Reinforcement Learning", "MLOps", "Graph ML", "Ethics in AI"],
    },
    "MECH": {
        "Sem 3": ["Thermodynamics", "Strength of Materials", "Manufacturing Processes", "Maths III", "Metrology"],
        "Sem 5": ["Heat Transfer", "Dynamics of Machines", "Design of Machine Elements", "Fluid Mechanics", "Manufacturing Lab"],
    },
    "Custom": {
        "My Semester": ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"]
    }
}

TAG_LIST = ["algo", "systems", "theory", "ml", "design"]

# ---------------------------------------------------------
# Synthetic dataset generator (training-only)
# Uses high-level features: subject_avg & low_count
# ---------------------------------------------------------
def make_synthetic_dataset(n=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    cgpa = rng.uniform(5.0, 9.8, n)
    attendance = rng.uniform(60, 100, n)
    study_hours = rng.uniform(2, 25, n)
    difficulty_pref = rng.integers(1, 5, n)   # 1..4
    backlogs = rng.integers(0, 5, n)

    subject_avg = rng.uniform(35, 95, n)      # mean of subject marks
    p_low = np.clip((55 - subject_avg) / 30, 0.01, 0.7)
    low_count = rng.binomial(6, p_low)        # approx count <40 out of ~6

    base = (
        0.55*cgpa
        + 0.02*attendance
        + 0.75*study_hours
        - 0.9*backlogs
        + 0.06*subject_avg
        - 0.35*low_count
        + 0.45*difficulty_pref
    )
    noise = rng.normal(0, 5, n)
    raw_score = base + noise
    grade_point = np.clip(raw_score / 3.5, 0, 10)
    pass_label = (grade_point >= 4.0).astype(int)

    return pd.DataFrame({
        "cgpa": cgpa,
        "attendance": attendance,
        "study_hours": study_hours,
        "backlogs": backlogs,
        "difficulty_pref": difficulty_pref,
        "subject_avg": subject_avg,
        "low_count": low_count,
        "grade_point": grade_point,
        "pass_label": pass_label
    })

# ---------------------------------------------------------
# Course catalog + helpers for recommender
# ---------------------------------------------------------
COURSE_CATALOG = pd.DataFrame([
    ("Data Structures",          3, {"algo":1, "systems":1, "theory":1, "ml":0, "design":0}, 5),
    ("Algorithms",               4, {"algo":1, "systems":0, "theory":1, "ml":0, "design":0}, 4),
    ("Database Systems",         3, {"algo":0, "systems":1, "theory":0, "ml":0, "design":1}, 5),
    ("Operating Systems",        4, {"algo":0, "systems":1, "theory":0, "ml":0, "design":0}, 4),
    ("Computer Networks",        3, {"algo":0, "systems":1, "theory":0, "ml":0, "design":0}, 4),
    ("Software Engineering",     2, {"algo":0, "systems":0, "theory":0, "ml":0, "design":1}, 5),
    ("Machine Learning",         4, {"algo":0, "systems":0, "theory":0, "ml":1, "design":0}, 5),
    ("Deep Learning",            5, {"algo":0, "systems":0, "theory":0, "ml":1, "design":0}, 4),
    ("Discrete Mathematics",     3, {"algo":1, "systems":0, "theory":1, "ml":0, "design":0}, 3),
    ("Cloud Computing",          3, {"algo":0, "systems":1, "theory":0, "ml":0, "design":0}, 4),
    ("UI/UX Design",             2, {"algo":0, "systems":0, "theory":0, "ml":0, "design":1}, 4),
    ("Data Visualization",       2, {"algo":0, "systems":0, "theory":0, "ml":0, "design":1}, 4),
], columns=["course", "difficulty", "tags", "popularity"])

def tag_vector(d): 
    return np.array([d.get(k, 0) for k in TAG_LIST], dtype=float)

COURSE_CATALOG["tag_vec"] = COURSE_CATALOG["tags"].apply(tag_vector)

# ---------------------------------------------------------
# Train models once (cached)
# ---------------------------------------------------------
@st.cache_resource
def train_models():
    df = make_synthetic_dataset()
    X = df[["cgpa","attendance","study_hours","backlogs","difficulty_pref","subject_avg","low_count"]]
    y_clf = df["pass_label"]
    y_reg = df["grade_point"]

    clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    reg = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

    Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.2, random_state=7, stratify=y_clf)
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=7)

    clf.fit(Xtr, ytr)
    reg.fit(Xtr_r, ytr_r)

    prob = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, prob)
    acc = accuracy_score(yte, (prob>=0.5).astype(int))
    pred_gp = reg.predict(Xte_r)
    mae = mean_absolute_error(yte_r, pred_gp)

    return clf, reg, (auc, acc, mae)

# ---------------------------------------------------------
# Recommender
# ---------------------------------------------------------
def recommend_courses(user_vec, pass_prob, diff_pref, k=6):
    rows = []
    for _, row in COURSE_CATALOG.iterrows():
        course = row["course"]
        diff = row["difficulty"]
        tagv = row["tag_vec"]
        pop = row["popularity"]

        # cosine similarity
        if np.linalg.norm(user_vec)==0 or np.linalg.norm(tagv)==0:
            sim = 0.0
        else:
            sim = float(np.dot(user_vec, tagv) / (np.linalg.norm(user_vec)*np.linalg.norm(tagv)))

        penalty = 0.0
        if pass_prob < 0.55 and diff >= 4: penalty -= 0.35
        if diff_pref <= 2 and diff >= 4:   penalty -= 0.20

        pop_norm = (pop - 1) / 4.0
        score = 0.55*sim + 0.30*pop_norm + 0.15*(1 - abs(diff - (diff_pref+1))) + penalty
        rows.append((course, score, diff, sim, pop))

    recos = pd.DataFrame(rows, columns=["course","score","difficulty","similarity","popularity"])
    recos = recos.sort_values("score", ascending=False).head(k)
    reasons = []
    for _, r in recos.iterrows():
        why=[]
        if r["similarity"]>=0.4: why.append("matches your interests")
        if r["difficulty"] <= (diff_pref+1): why.append("within your difficulty comfort")
        if r["popularity"]>=4: why.append("popular with peers")
        if not why: why=["good overall fit"]
        reasons.append(", ".join(why))
    recos["why"] = reasons
    return recos

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Learning Path Advisor", page_icon="🎓", layout="wide")
st.title("🎓 Smart Learning Path Advisor — Branch/Semester & Per-Subject Marks")

# --- Student meta & interests (sidebar) ---
with st.sidebar:
    st.header("Student Profile")
    name = st.text_input("Student Name", "")
    regno = st.text_input("Register No.", "")
    cgpa = st.slider("Current CGPA", 5.0, 10.0, 7.2, 0.1)
    attendance = st.slider("Attendance (%)", 60, 100, 85, 1)
    study_hours = st.slider("Study Hours / week", 0, 30, 8, 1)
    backlogs = st.slider("Existing Backlogs (count)", 0, 6, 0, 1)
    diff_pref = st.select_slider("Difficulty Tolerance", options=[1,2,3,4], value=2)
    st.caption("1 = prefer easy, 4 = okay with hard")

    st.markdown("---")
    st.subheader("Interests (0–5)")
    i_algo    = st.slider("Algorithms/Theory", 0, 5, 3)
    i_systems = st.slider("Systems/Infra", 0, 5, 3)
    i_theory  = st.slider("Discrete/Theory", 0, 5, 2)
    i_ml      = st.slider("AI/ML", 0, 5, 4)
    i_design  = st.slider("Design/UX", 0, 5, 2)

# --- Branch & Semester selection ---
st.markdown("### Select Program & Semester")
colp, cols = st.columns(2)
with colp:
    branch = st.selectbox("Branch", list(SUBJECT_MAP.keys()), index=0)
with cols:
    semester = st.selectbox("Semester", list(SUBJECT_MAP[branch].keys()))

subjects_default = SUBJECT_MAP[branch][semester]

st.markdown("#### Subjects for this semester")
st.write(", ".join(subjects_default))

# --- Allow customization of subject list ---
customize = st.checkbox("I want to customize the subject list", value=(branch=="Custom"))
if customize:
    text_help = "Edit subjects as comma-separated names (e.g., Maths, DBMS, OS)"
    subj_text = st.text_area("Subjects", value=", ".join(subjects_default), help=text_help)
    subjects = [s.strip() for s in subj_text.split(",") if s.strip()]
else:
    subjects = subjects_default

# --- Per-subject marks inputs ---
st.markdown("### Enter marks per subject (/100)")
grid_cols = st.columns(3)
marks_dict = {}
for i, sub in enumerate(subjects):
    with grid_cols[i % 3]:
        marks_dict[sub] = st.number_input(f"{sub}", min_value=0, max_value=100, value=75, step=1, key=f"mark_{i}")

# Derived features from marks
marks = np.array(list(marks_dict.values()), dtype=float) if marks_dict else np.array([])
subject_avg = float(np.mean(marks)) if marks.size else 0.0
low_count = int(np.sum(marks < 40)) if marks.size else 0
subject_std = float(np.std(marks)) if marks.size else 0.0

with st.expander("Subject summary", expanded=True):
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Subjects", f"{len(subjects)}")
    colB.metric("Average mark", f"{subject_avg:.1f}/100")
    colC.metric("Low scores (<40)", f"{low_count}")
    colD.metric("Variation (std dev)", f"{subject_std:.1f}")

st.markdown("---")

# Train/load models
clf, reg, (auc, acc, mae) = train_models()

# Predict
feat = np.array([[cgpa, attendance, study_hours, backlogs, diff_pref, subject_avg, low_count]])
pass_prob = float(clf.predict_proba(feat)[0,1])
gp_pred = float(np.clip(reg.predict(feat)[0], 0, 10))

def gp_to_letter(gp):
    if gp >= 8.5: return "A"
    if gp >= 6.5: return "B"
    if gp >= 4.0: return "C"
    return "F"

col1, col2, col3 = st.columns(3)
with col1: st.metric("Pass Probability", f"{pass_prob*100:.1f}%")
with col2: st.metric("Expected Grade Point (0–10)", f"{gp_pred:.2f}")
with col3: st.metric("Expected Letter Grade", gp_to_letter(gp_pred))

st.markdown("### Why this prediction?")
try:
    lr = clf.named_steps["lr"]
    scaler = clf.named_steps["scaler"]
    names = ["cgpa","attendance","study_hours","backlogs","difficulty_pref","subject_avg","low_count"]
    z = scaler.transform(feat)[0]
    contrib = z * lr.coef_[0]
    top_ix = np.argsort(np.abs(contrib))[::-1][:5]
    expl = pd.DataFrame({
        "feature":[names[i] for i in top_ix],
        "contribution":[float(contrib[i]) for i in top_ix],
        "effect":["↑ increases pass" if contrib[i]>0 else "↓ decreases pass" for i in top_ix]
    })
    st.dataframe(expl, use_container_width=True)
except Exception:
    st.caption("Explanation unavailable for this model.")

st.markdown("### Course Recommendations")
user_interest_vec = np.array([i_algo, i_systems, i_theory, i_ml, i_design], dtype=float)
recos = recommend_courses(user_interest_vec, pass_prob, diff_pref, k=6)
st.dataframe(recos[["course","score","difficulty","why"]].reset_index(drop=True), use_container_width=True)

# Optional: one-click CSV of this record (useful for demo)
row = {
    "name": name, "regno": regno, "branch": branch, "semester": semester,
    "cgpa": cgpa, "attendance": attendance, "study_hours": study_hours,
    "backlogs": backlogs, "diff_pref": diff_pref, "subject_avg": subject_avg,
    "low_count": low_count, "pass_probability": round(pass_prob,4),
    "expected_grade_point": round(gp_pred,2), "letter_grade": gp_to_letter(gp_pred)
}
download_df = pd.DataFrame([row])
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download this result as CSV", data=csv_bytes, file_name="smart_learning_result.csv", mime="text/csv")

with st.expander("Model sanity metrics (synthetic test)"):
    st.write(f"AUC: **{auc:.3f}**, Accuracy: **{acc:.3f}**")
    st.write(f"Regression MAE (grade point): **{mae:.2f}**")

st.markdown("---")
st.caption("MVP: Pick branch/semester → subjects auto-load (editable). Enter marks → AI predicts pass risk & grade and recommends courses. Replace synthetic training with your real data later.")
