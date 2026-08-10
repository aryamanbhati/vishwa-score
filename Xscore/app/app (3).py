import streamlit as st
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config
from voice_advisor import run_asr, generate_response, text_to_speech, LANG_NAMES
st.set_page_config(
    page_title="Vishwa Score
    core — Alternative Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.pillar-bar{height:8px;border-radius:4px;background:#e8e8e8;margin-top:4px}
.pillar-fill{height:8px;border-radius:4px}
</style>
""", unsafe_allow_html=True)

cfg = Config()

@st.cache_resource(show_spinner=False)
def get_conn(http_path):
    host = cfg.host.replace("https://", "").replace("http://", "")
    return sql.connect(
        server_hostname=host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
        _use_arrow_native_complex_types=False,
    )

@st.cache_data(ttl=300, show_spinner=False)
def q(http_path, statement):
    try:
        conn   = get_conn(http_path)
        cursor = conn.cursor()
        cursor.execute(statement)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Vishwa Score")
    st.markdown("**Alternative Credit Scoring**")
    st.markdown("*Bharat Bricks Hacks 2026*")
    st.divider()

    http_path = st.text_input(
        "SQL Warehouse HTTP Path",
        placeholder="/sql/1.0/warehouses/abc123",
        help="SQL Warehouses → your warehouse → Connection Details → HTTP Path"
    )

    if not http_path:
        st.info("Enter your SQL Warehouse HTTP Path above to load data.")
        st.stop()

    page = st.radio(
        "Navigation",
        ["Portfolio Overview",
         "Segment Analytics",
         "User Lookup",
         "Model Performance",
         "Voice Advisor"],
        label_visibility="collapsed",
    )

    st.divider()
    total = q(http_path, "SELECT COUNT(*) AS n FROM xscore.gold.credit_scores")
    if not total.empty:
        st.metric("Users scored", f"{int(total['n'].iloc[0]):,}")
    avg_d = q(http_path,
        "SELECT ROUND(AVG(default_probability)*100,2) AS v "
        "FROM xscore.gold.credit_scores")
    if not avg_d.empty:
        st.metric("Avg default risk", f"{float(avg_d['v'].iloc[0]):.2f}%")
    st.caption("Cache refreshes every 5 min")

# ── Helpers ───────────────────────────────────────────────────
BAND_COLOR = {"Excellent":"#1D9E75","Good":"#378ADD","Fair":"#BA7517","Poor":"#E24B4A"}

def bar(pct, color):
    return (f"<div class='pillar-bar'>"
            f"<div class='pillar-fill' style='width:{min(100,abs(pct)):.0f}%;"
            f"background:{color}'></div></div>")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption("Live from Vishwa score.gold.credit_scores")

    kpi = q(http_path, """
        SELECT COUNT(*) AS total,
               ROUND(AVG(xscore),0) AS avg_score,
               ROUND(AVG(default_probability)*100,2) AS avg_def,
               SUM(CASE WHEN score_band='Excellent' THEN 1 ELSE 0 END) AS exc,
               SUM(CASE WHEN score_band='Good'      THEN 1 ELSE 0 END) AS good
        FROM xscore.gold.credit_scores
    """)
    if not kpi.empty:
        r = kpi.iloc[0]
        total_n  = int(r["total"])
        lendable = int(r["exc"] or 0) + int(r["good"] or 0)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total users",         f"{total_n:,}")
        c2.metric("Average Vishwa Score",      f"{int(float(r['avg_score']))}")
        c3.metric("Avg default risk",    f"{float(r['avg_def']):.2f}%")
        c4.metric("Lendable (Good+Exc)", f"{lendable:,}",
                  delta=f"{lendable/total_n*100:.0f}% of portfolio")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Score bands")
        band = q(http_path, """
            SELECT score_band,
                   COUNT(*) AS users,
                   ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),1) AS pct,
                   ROUND(AVG(xscore),0) AS avg_score,
                   ROUND(AVG(default_probability)*100,2) AS def_pct
            FROM xscore.gold.credit_scores
            GROUP BY score_band ORDER BY avg_score DESC
        """)
        if not band.empty:
            for _, row in band.iterrows():
                c   = BAND_COLOR.get(str(row["score_band"]), "#888")
                pct = float(row["pct"])
                st.markdown(
                    f"<div style='margin-bottom:12px'>"
                    f"<span style='font-weight:700;color:{c}'>{row['score_band']}</span>"
                    f" &nbsp; {int(float(row['users'])):,} users &nbsp;·&nbsp; "
                    f"{pct:.1f}% &nbsp;·&nbsp; avg {int(float(row['avg_score']))} "
                    f"&nbsp;·&nbsp; {float(row['def_pct']):.2f}% default"
                    f"{bar(pct, c)}</div>",
                    unsafe_allow_html=True,
                )

    with col_r:
        st.subheader("Score histogram")
        hist = q(http_path, """
            SELECT FLOOR(xscore/50)*50 AS bucket, COUNT(*) AS users
            FROM xscore.gold.credit_scores
            GROUP BY bucket ORDER BY bucket
        """)
        if not hist.empty:
            hist["bucket"] = hist["bucket"].astype(float).astype(int)
            hist["users"]  = hist["users"].astype(int)
            st.bar_chart(hist.set_index("bucket")["users"],
                         use_container_width=True, height=280)

    st.divider()
    st.subheader("Score vs default probability")
    scatter = q(http_path, """
        SELECT xscore, ROUND(default_probability*100,2) AS default_pct,
               score_band, segment
        FROM xscore.gold.credit_scores ORDER BY RAND() LIMIT 1500
    """)
    if not scatter.empty:
        scatter["xscore"]      = scatter["xscore"].astype(float)
        scatter["default_pct"] = scatter["default_pct"].astype(float)
        st.scatter_chart(scatter, x="xscore", y="default_pct",
                         color="score_band",
                         use_container_width=True, height=320)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT ANALYTICS
# ════════════════════════════════════════════════════════════════
elif page == "Segment Analytics":
    st.title("Segment Analytics")
    st.caption("All 6 pillars broken down by borrower segment")

    seg = q(http_path, """
        SELECT c.segment,
               COUNT(*) AS users,
               ROUND(AVG(c.xscore),0) AS avg_score,
               ROUND(AVG(c.default_probability)*100,2) AS def_pct,
               ROUND(AVG(e.shap_pillar_1__bill_payment_pts),1) AS p1,
               ROUND(AVG(e.shap_pillar_2__upi_flow_pts),1)     AS p2,
               ROUND(AVG(e.shap_pillar_3__assets_pts),1)       AS p3,
               ROUND(AVG(e.shap_pillar_4__income_pts),1)       AS p4,
               ROUND(AVG(e.shap_pillar_5__identity_pts),1)     AS p5,
               ROUND(AVG(e.shap_pillar_6__stability_pts),1)    AS p6
        FROM xscore.gold.credit_scores c
        JOIN xscore.gold.score_explanations e USING (user_id)
        GROUP BY c.segment ORDER BY avg_score DESC
    """)

    if not seg.empty:
        PNAMES  = ["Bills","UPI","Assets","Income","Identity","Stability"]
        PCOLS   = ["p1","p2","p3","p4","p5","p6"]
        PCOLORS = ["#1D9E75","#378ADD","#BA7517","#534AB7","#D4537E","#D85A30"]

        for _, row in seg.iterrows():
            name  = str(row["segment"]).replace("_"," ").title()
            score = int(float(row["avg_score"]))
            users = int(float(row["users"]))
            with st.expander(
                f"**{name}** — avg score {score} · "
                f"{users:,} users · {float(row['def_pct']):.1f}% default",
                expanded=(row["segment"]=="kirana_owner"),
            ):
                cols = st.columns(6)
                for col, nm, pc, color in zip(cols, PNAMES, PCOLS, PCOLORS):
                    pts  = float(row[pc] or 0)
                    sign = "+" if pts >= 0 else ""
                    c    = color if pts >= 0 else "#E24B4A"
                    with col:
                        st.markdown(
                            f"<div style='text-align:center;padding:6px'>"
                            f"<div style='font-size:11px;color:#888'>{nm}</div>"
                            f"<div style='font-size:22px;font-weight:700;color:{c}'>"
                            f"{sign}{pts:.0f}</div>"
                            f"<div style='font-size:10px;color:#aaa'>pts</div></div>",
                            unsafe_allow_html=True,
                        )

        st.divider()
        st.subheader("Pillar heatmap")
        seg["segment"] = seg["segment"].str.replace("_"," ").str.title()
        heat = seg.set_index("segment")[PCOLS].rename(
            columns=dict(zip(PCOLS, PNAMES))
        ).astype(float)
        st.dataframe(
            heat.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}"),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════
# PAGE 3 — USER LOOKUP
# ════════════════════════════════════════════════════════════════
elif page == "User Lookup":
    st.title("User Lookup")

    c1, c2 = st.columns([4,1])
    with c1:
        uid = st.text_input("User ID", value="USR000001",
                            placeholder="USR000001 … USR050000")
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Random", use_container_width=True):
            rand = q(http_path,
                "SELECT user_id FROM xscore.gold.credit_scores "
                "ORDER BY RAND() LIMIT 1")
            if not rand.empty:
                uid = rand["user_id"].iloc[0]
                st.rerun()

    if uid:
        row_df = q(http_path, f"""
            SELECT c.user_id, c.segment, c.xscore, c.score_band,
                   ROUND(c.default_probability*100,2) AS def_pct,
                   c.score_timestamp, c.model_version,
                   e.explanation_text,
                   e.top_positive_factors, e.top_negative_factors,
                   ROUND(e.shap_pillar_1__bill_payment_pts,1) AS p1,
                   ROUND(e.shap_pillar_2__upi_flow_pts,1)     AS p2,
                   ROUND(e.shap_pillar_3__assets_pts,1)       AS p3,
                   ROUND(e.shap_pillar_4__income_pts,1)       AS p4,
                   ROUND(e.shap_pillar_5__identity_pts,1)     AS p5,
                   ROUND(e.shap_pillar_6__stability_pts,1)    AS p6
            FROM xscore.gold.credit_scores c
            LEFT JOIN xscore.gold.score_explanations e USING (user_id)
            WHERE c.user_id = '{uid}'
        """)

        if row_df.empty:
            st.warning(f"User `{uid}` not found. Try USR000001 – USR050000")
        else:
            r     = row_df.iloc[0]
            score = int(float(r["xscore"] or 500))
            band  = str(r["score_band"])
            color = BAND_COLOR.get(band, "#888")

            st.markdown(f"""
            <div style='background:#f8f9fa;border-radius:12px;padding:24px;
                        border-left:6px solid {color};margin-bottom:20px'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='flex:1'>
                  <div style='font-size:12px;color:#888;margin-bottom:6px'>
                    {r['user_id']} &nbsp;·&nbsp;
                    {str(r['segment']).replace('_',' ').title()}
                  </div>
                  <div style='font-size:15px;color:#333;margin-bottom:10px;
                              max-width:500px;line-height:1.6'>
                    {str(r.get('explanation_text',''))}
                  </div>
                  <div style='font-size:11px;color:#aaa'>
                    Scored: {str(r.get('score_timestamp',''))[:19]}
                    &nbsp;·&nbsp; {str(r.get('model_version',''))}
                  </div>
                </div>
                <div style='text-align:right;padding-left:24px'>
                  <div style='font-size:11px;color:#888'>XScore</div>
                  <div style='font-size:56px;font-weight:700;
                              color:{color};line-height:1.1'>{score}</div>
                  <div style='font-size:14px;font-weight:600;color:{color}'>{band}</div>
                  <div style='font-size:12px;color:#888;tmargin-top:4px'>
                    {float(r['def_pct']):.1f}% default risk
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("6-Pillar Breakdown")
            PILLARS = [
                ("Pillar 1","Bill Payment",  float(r["p1"] or 0),"#1D9E75",
                 "On-time rate · Late days · Payment trend"),
                ("Pillar 2","UPI & Digital", float(r["p2"] or 0),"#378ADD",
                 "Txn volume · Merchant diversity · Failure rate"),
                ("Pillar 3","Assets",        float(r["p3"] or 0),"#BA7517",
                 "Land · Vehicle · Bank tenure · FD/RD"),
                ("Pillar 4","Income",        float(r["p4"] or 0),"#534AB7",
                 "Income level · ITR · GST · Employment"),
                ("Pillar 5","Identity",      float(r["p5"] or 0),"#D4537E",
                 "Jan Dhan · SHG · DBT · SVANidhi"),
                ("Pillar 6","Stability",     float(r["p6"] or 0),"#D85A30",
                 "SIM tenure · Location · KYC · Fraud"),
            ]
            for label, name, pts, col, desc in PILLARS:
                bc   = col if pts >= 0 else "#E24B4A"
                bpct = min(100, abs(pts)/150*100)
                sign = "+" if pts >= 0 else ""
                cl, cr = st.columns([2,5])
                with cl:
                    st.markdown(
                        f"<div style='padding:8px 0'>"
                        f"<div style='font-size:11px;color:#888'>{label}</div>"
                        f"<div style='font-size:13px;font-weight:600'>{name}</div>"
                        f"<div style='font-size:22px;font-weight:700;color:{bc}'>"
                        f"{sign}{pts:.0f} pts</div></div>",
                        unsafe_allow_html=True,
                    )
                with cr:
                    st.markdown(
                        f"<div style='padding-top:20px;font-size:11px;color:#aaa'>{desc}</div>"
                        f"<div class='pillar-bar'>"
                        f"<div class='pillar-fill' style='width:{bpct:.0f}%;background:{bc}'>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            st.divider()
            cp, cn = st.columns(2)
            with cp:
                st.markdown("**Top positive factors**")
                for f in str(r.get("top_positive_factors","")).split(" | "):
                    if f.strip() and "No strong" not in f:
                        st.success(f"✓  {f.strip()}")
            with cn:
                st.markdown("**Top negative factors**")
                for f in str(r.get("top_negative_factors","")).split(" | "):
                    if f.strip() and "No strong" not in f:
                        st.error(f"✗  {f.strip()}")

            st.divider()
            st.subheader("Bank Recommendation")
            def_pct = float(r["def_pct"])
            if score >= 700:
                st.success(f"**RECOMMEND APPROVAL** — Score {score}. "
                           f"Default risk {def_pct:.1f}%. Eligible for standard products.")
            elif score >= 550:
                st.warning(f"**CONDITIONAL APPROVAL** — Score {score}. "
                           f"Moderate risk {def_pct:.1f}%. Consider smaller amount or collateral.")
            else:
                st.error(f"**HIGH RISK** — Score {score}. "
                         f"Default risk {def_pct:.1f}%. Advise building payment history first.")


# ════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("The XScore thesis in numbers")

    st.subheader("Progressive AUC improvement")
    st.info("Adding alternative data (bills, UPI, govt signals) lifts AUC by +0.12 "
            "vs income/employment baseline. None of this data exists in CIBIL.")

    thesis = pd.DataFrame({
        "Version"    : ["v1 Baseline","v2 + Bills","v3 Full XScore"],
        "Features"   : [7, 12, 32],
        "Data added" : ["Income + employment",
                        "+ Utility/mobile bill history",
                        "+ UPI + govt + stability"],
        "Val AUC"    : ["~0.74","~0.80","~0.86"],
        "Lift"       : ["—","+0.06","+0.12 total"],
        "In CIBIL?"  : ["✓ Yes","✗ No","✗ No"],
    })
    st.dataframe(thesis, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Hyperopt tuning history")
    hopt = q(http_path, """
        SELECT CAST(run_timestamp AS STRING) AS run_time,
               n_trials,
               ROUND(best_auc,4) AS best_auc,
               ROUND(champion_auc,4) AS prev_champion,
               ROUND(improvement,4) AS improvement,
               CAST(promoted AS STRING) AS promoted
        FROM xscore.gold.best_hyperparams
        ORDER BY run_timestamp DESC LIMIT 10
    """)
    if not hopt.empty:
        st.dataframe(hopt, use_container_width=True, hide_index=True)
    else:
        st.caption("No Hyperopt runs yet — run NB_06.")

    st.divider()
    st.subheader("Pillar importance (avg |SHAP|)")
    imp = q(http_path, """
        SELECT
            ROUND(AVG(ABS(shap_pillar_1__bill_payment_pts)),1) AS Bills,
            ROUND(AVG(ABS(shap_pillar_2__upi_flow_pts)),1)     AS UPI,
            ROUND(AVG(ABS(shap_pillar_3__assets_pts)),1)       AS Assets,
            ROUND(AVG(ABS(shap_pillar_4__income_pts)),1)       AS Income,
            ROUND(AVG(ABS(shap_pillar_5__identity_pts)),1)     AS Identity,
            ROUND(AVG(ABS(shap_pillar_6__stability_pts)),1)    AS Stability
        FROM xscore.gold.score_explanations
    """)
    if not imp.empty:
        r   = imp.iloc[0]
        idf = pd.DataFrame({
            "Pillar": list(r.index),
            "Avg |SHAP| pts": [float(v) for v in r.values],
        }).sort_values("Avg |SHAP| pts", ascending=False)
        st.bar_chart(idf.set_index("Pillar"), use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════
# PAGE 5 — VOICE ADVISOR (ArthaSetu)
# ════════════════════════════════════════════════════════════════
"""
XScore + ArthaSetu — Integrated Dashboard
==========================================
Pages: Portfolio Overview | Segment Analytics | User Lookup | Model Performance | Voice Advisor

pip install streamlit pandas databricks-sql-connector databricks-sdk requests
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import requests
import re
import base64
import tempfile
from databricks import sql
from databricks.sdk.core import Config

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SARVAM AI CONFIG + FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SARVAM_KEY = "REVOKED_SARVAM_KEY_SEE_ENV"

LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
    "en": "English", "kn": "Kannada", "ml": "Malayalam",
}
LANG_CODE = {k: f"{k}-IN" for k in LANG_NAMES}
LANG_SPEAKER = {
    "hi": "anushka", "ta": "arya", "te": "manisha", "mr": "vidya",
    "bn": "karun", "gu": "hitesh", "pa": "abhilash",
    "kn": "anushka", "ml": "arya", "en": "anushka",
}

SCHEMES_CONTEXT = """
GOVERNMENT LOAN SCHEMES FOR RURAL INDIA:

1. PM SVANidhi Tier 1: Street vendors/hawkers. Loan up to Rs 10,000, 7% interest, no collateral. Need vending certificate + Aadhaar.
2. PM SVANidhi Tier 2: Vendors who repaid Tier 1. Up to Rs 20,000, 7%, no collateral.
3. PM SVANidhi Tier 3: Vendors who repaid Tier 1+2. Up to Rs 50,000, 7%, no collateral.
4. PMMY Mudra Shishu: Micro enterprises, small shops. Up to Rs 50,000, ~10%, no collateral. Need Aadhaar + PAN.
5. PMMY Mudra Kishor: Growing businesses. Rs 50,000-5,00,000, ~12%, partial collateral. Need 1yr business vintage.
6. PMMY Mudra Tarun: Established businesses. Rs 5,00,000-10,00,000, ~14%, collateral required. Need 2yr vintage + financials.
7. Kisan Credit Card (Crop): Farmers. Up to Rs 3,00,000, 7% (3% subvention), need land records.
8. Kisan Credit Card (Allied): Dairy/poultry/fishery. Up to Rs 2,00,000, 7%, need land records.
9. NABARD Dairy Development: Dairy farming. Up to Rs 7,00,000, 8.5%, 25% subsidy (33% SC/ST).
10. SHG Bank Linkage: Women self-help groups. Up to Rs 10,00,000, 7%, no collateral. Need active SHG 6 months.
11. PM Vishwakarma Tier 1: Artisans (tailor/carpenter/weaver/potter/barber). Up to Rs 1,00,000, 5%, no collateral.
12. PM Vishwakarma Tier 2: Artisans who repaid Tier 1. Up to Rs 2,00,000, 5%, no collateral.
"""
