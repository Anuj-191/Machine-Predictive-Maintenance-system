"""
app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit Web Application for SC08 — Machine Predictive Maintenance & Fault Priority System.

Hybrid Artificial Neural Network (ANN) + Mamdani Fuzzy Logic Decision System
Built strictly using NumPy and pre-trained weights/normalisation parameters.
"""

import os
import sys

# Ensure repository root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.prediction import PredictiveMaintenancePipeline


# ==============================================================================
# Page Configuration & Industrial CSS Theme
# ==============================================================================

st.set_page_config(
    page_title="SC08 Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text-color, #0F172A);
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--text-color, #475569);
        opacity: 0.85;
        margin-bottom: 1.25rem;
    }
    .priority-badge-LOW {
        background-color: #16A34A;
        color: #FFFFFF;
        padding: 0.35rem 1.1rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #15803D;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .priority-badge-MEDIUM {
        background-color: #EAB308;
        color: #0F172A;
        padding: 0.35rem 1.1rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #CA8A04;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .priority-badge-HIGH {
        background-color: #EA580C;
        color: #FFFFFF;
        padding: 0.35rem 1.1rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #C2410C;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .priority-badge-URGENT {
        background-color: #DC2626;
        color: #FFFFFF;
        padding: 0.35rem 1.1rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #991B1B;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .recommendation-box {
        background-color: var(--secondary-background-color, rgba(14, 165, 233, 0.08));
        border: 1px solid rgba(14, 165, 233, 0.35);
        border-left: 5px solid #0EA5E9;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        font-size: 0.98rem;
        color: var(--text-color, inherit);
    }
    .decision-banner {
        padding: 0.85rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.0rem;
        margin-bottom: 1rem;
    }
    .decision-failure {
        background-color: rgba(239, 68, 68, 0.14);
        border: 1px solid #EF4444;
        color: var(--text-color, inherit);
    }
    .decision-healthy {
        background-color: rgba(34, 197, 94, 0.14);
        border: 1px solid #22C55E;
        color: var(--text-color, inherit);
    }
    .flow-step-box {
        background-color: var(--secondary-background-color, rgba(241, 245, 249, 0.7));
        border: 1px solid var(--border-color, rgba(148, 163, 184, 0.35));
        padding: 8px 14px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        color: var(--text-color, inherit);
        font-size: 0.9rem;
        margin: 3px 0;
    }
    .flow-arrow {
        text-align: center;
        color: #0EA5E9;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 1px 0;
    }
    .footer-text {
        text-align: center;
        color: var(--text-color, #64748B);
        opacity: 0.75;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cached Pipeline Loader (Load Weights & Normalisation Params Once)
# ==============================================================================

@st.cache_resource
def load_pipeline():
    """
    Instantiate and cache the PredictiveMaintenancePipeline.
    Ensures pre-trained weights (ann_weights.npz) and normalisation parameters
    (norm_params.json) are loaded into memory exactly once.
    """
    return PredictiveMaintenancePipeline()


# ==============================================================================
# Benchmark Test Scenarios (All within strict Fuzzy Universe Limits)
# ==============================================================================

PRESET_SCENARIOS = {
    "Custom User Input": None,
    "Scenario 1: Normal Machine (Low Risk)": {
        "product_type": "L",
        "air_temp_K": 300.0,
        "process_temp_K": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_Nm": 40.0,
        "tool_wear_min": 20.0,
    },
    "Scenario 2: Moderate Risk Machine": {
        "product_type": "M",
        "air_temp_K": 301.0,
        "process_temp_K": 311.0,
        "rotational_speed_rpm": 1400.0,
        "torque_Nm": 48.0,
        "tool_wear_min": 130.0,
    },
    "Scenario 3: High ANN Probability (Elevated Risk)": {
        "product_type": "H",
        "air_temp_K": 303.0,
        "process_temp_K": 313.0,
        "rotational_speed_rpm": 1200.0,
        "torque_Nm": 65.0,
        "tool_wear_min": 50.0,
    },
    "Scenario 4: High Tool Wear Machine": {
        "product_type": "L",
        "air_temp_K": 299.0,
        "process_temp_K": 309.0,
        "rotational_speed_rpm": 1550.0,
        "torque_Nm": 38.0,
        "tool_wear_min": 225.0,
    },
    "Scenario 5: Critical Emergency Condition": {
        "product_type": "M",
        "air_temp_K": 304.0,
        "process_temp_K": 314.0,
        "rotational_speed_rpm": 1180.0,
        "torque_Nm": 68.0,
        "tool_wear_min": 235.0,
    },
}


# ==============================================================================
# Main Streamlit Application
# ==============================================================================

def main():
    # Header Section
    st.markdown('<div class="main-header">SC08 — Machine Predictive Maintenance & Fault Priority System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hybrid Artificial Neural Network (ANN) + Mamdani Fuzzy Logic Decision System (Product V1)</div>', unsafe_allow_html=True)

    # 1. Pipeline Loading
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load predictive maintenance pipeline: {e}")
        st.stop()

    # 2. Sidebar Input Form
    st.sidebar.header("⚙️ Machine Operating Parameters")

    preset_choice = st.sidebar.selectbox(
        "Load Test Scenario Preset",
        options=list(PRESET_SCENARIOS.keys()),
        index=0,
        help="Quickly populate inputs with validated operating scenarios."
    )

    preset_data = PRESET_SCENARIOS[preset_choice]

    # Boundary notice explaining the fuzzy limit constraints
    st.sidebar.info(
        "ℹ️ **Universe Boundary Limits:**\n"
        "- **Tool wear** is capped at **260 min** (fuzzy universe [0, 260] min).\n"
        "- **Torque** is capped at **80 Nm** (fuzzy universe [0, 80] Nm).\n"
        "Inputs above these limits are restricted in the UI to keep crisp readings strictly within the defined fuzzy domain."
    )

    with st.sidebar.form("input_form"):
        st.subheader("Raw Sensor Readings")

        product_type = st.selectbox(
            "Product Type (Quality Variant)",
            options=["L", "M", "H"],
            index=["L", "M", "H"].index(preset_data["product_type"]) if preset_data else 0,
            help="Quality variant: L (Low, 50%), M (Medium, 30%), H (High, 20%). L serves as reference category."
        )

        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=float(preset_data["air_temp_K"]) if preset_data else 300.0,
            step=0.5,
            format="%.1f",
            help="Ambient room / intake air temperature in Kelvin [250.0, 350.0 K]."
        )

        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=float(preset_data["process_temp_K"]) if preset_data else 310.0,
            step=0.5,
            format="%.1f",
            help="Machine cutting process temperature in Kelvin [250.0, 350.0 K]."
        )

        speed_rpm = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0.0,
            max_value=5000.0,
            value=float(preset_data["rotational_speed_rpm"]) if preset_data else 1500.0,
            step=10.0,
            format="%.1f",
            help="Spindle rotational speed in revolutions per minute [0.0, 5000.0 rpm]."
        )

        # STRICT BOUNDARY: Max 80.0 Nm for torque
        torque_nm = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=80.0,
            value=float(preset_data["torque_Nm"]) if preset_data else 40.0,
            step=0.5,
            format="%.1f",
            help="Spindle load torque in Newton-metres [0.0, 80.0 Nm]. Capped at 80.0 Nm to align with the fuzzy inference universe."
        )

        # STRICT BOUNDARY: Max 260.0 min for tool wear
        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0.0,
            max_value=260.0,
            value=float(preset_data["tool_wear_min"]) if preset_data else 20.0,
            step=1.0,
            format="%.1f",
            help="Cumulative tool usage in minutes [0.0, 260.0 min]. Capped at 260.0 min to align with the fuzzy inference universe."
        )

        submit_btn = st.form_submit_button("🔍 Run Prediction Pipeline", type="primary", use_container_width=True)

    # 3. Form Submission & Single Pipeline Call
    inputs_to_run = {
        "product_type": product_type,
        "air_temp_K": air_temp,
        "process_temp_K": process_temp,
        "rotational_speed_rpm": speed_rpm,
        "torque_Nm": torque_nm,
        "tool_wear_min": tool_wear,
    }

    # Boundary enforcement check
    validation_error = None
    if torque_nm > 80.0:
        validation_error = f"Torque value ({torque_nm} Nm) exceeds the maximum supported fuzzy universe limit of 80.0 Nm."
    elif tool_wear > 260.0:
        validation_error = f"Tool wear value ({tool_wear} min) exceeds the maximum supported fuzzy universe limit of 260.0 min."

    result = None
    if validation_error:
        st.error(f"Input Validation Error: {validation_error}")
    else:
        try:
            # Single execution of validated pipeline without duplicating any inference logic
            result = pipeline.predict(**inputs_to_run)
        except (ValueError, TypeError) as e:
            st.error(f"Input Validation Error: {e}")
        except Exception as e:
            st.error(f"Prediction Pipeline Error: {e}")

    # 4. Display Results
    if result is not None:
        fault_prob = result["fault_probability"]
        severity_index = result["severity_index"]
        priority = result["priority"]
        recommendation = result["recommendation"]
        fuzzy_explanation = result["fuzzy_explanation"]
        fired_rules = result["fired_rules"]

        st.subheader("📊 Live Machine Assessment Results")

        # ── ANN THRESHOLD DECISION BANNER (SEPARATE FROM FUZZY PRIORITY) ───────
        st.markdown("##### 1. Artificial Neural Network — Binary Threshold Decision")
        if fault_prob >= 0.50:
            st.markdown(
                f'<div class="decision-banner decision-failure">'
                f'⚠️ <strong>ANN Threshold Decision: FAILURE PREDICTED</strong> &bull; '
                f'P(Failure) = {fault_prob * 100:.2f}% (Prob = {fault_prob:.4f} &ge; 0.50 decision threshold).<br>'
                f'<span style="font-size:0.9rem; font-weight:normal;">'
                f'The neural network flags this operational sensor state as high-risk / anomalous. '
                f'This binary decision is derived strictly from the ANN classification threshold (&tau; = 0.50), '
                f'evaluated independently of the physical fuzzy severity below.'
                f'</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="decision-banner decision-healthy">'
                f'✅ <strong>ANN Threshold Decision: MACHINE HEALTHY</strong> &bull; '
                f'P(Failure) = {fault_prob * 100:.2f}% (Prob = {fault_prob:.4f} &lt; 0.50 decision threshold).<br>'
                f'<span style="font-size:0.9rem; font-weight:normal;">'
                f'The neural network predicts normal operation within expected parameter variance under the standard &tau; = 0.50 threshold.'
                f'</span></div>',
                unsafe_allow_html=True
            )

        # ── CORE ASSESSMENT METRIC CARDS ──────────────────────────────────────
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric(
                label="ANN Fault Probability P(Failure)",
                value=f"{fault_prob * 100:.2f}%",
                delta=f"Raw Prob: {fault_prob:.4f}"
            )

        with col_m2:
            st.metric(
                label="Mamdani Fuzzy Severity Index",
                value=f"{severity_index:.2f} / 100",
                delta=f"Priority: {priority}"
            )

        with col_m3:
            st.write("**Maintenance Priority:**")
            st.markdown(f"<span class='priority-badge-{priority}'>{priority}</span>", unsafe_allow_html=True)

        # ── SEVERITY GAUGE PROGRESS BAR ───────────────────────────────────────
        st.write("**Continuous Severity Scale (0–100):**")
        sev_fraction = min(1.0, max(0.0, severity_index / 100.0))
        st.progress(sev_fraction)

        # ── RECOMMENDED MAINTENANCE ACTION ────────────────────────────────────
        st.markdown(f"""
            <div class="recommendation-box">
                <strong>🛠️ Actionable Maintenance Recommendation:</strong><br>
                {recommendation}
            </div>
        """, unsafe_allow_html=True)

        # ── EXPLAINABILITY & SYNTHESIS ────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Decision Explainability & Rule Activations")
        st.info(f"**Synthesis:** {fuzzy_explanation}")

        # Expandable section for fired fuzzy rules
        with st.expander(f"🔍 Fired Fuzzy Rules Detail ({len(fired_rules)} rule(s) activated)", expanded=True):
            if fired_rules:
                for r in fired_rules:
                    st.markdown(
                        f"- **Rule R{r['rule']}** `[{r['consequent']} @ strength {r['strength']:.4f}]`: {r['description']}"
                    )
            else:
                st.write("*(No specific linguistic rules fired — safe fallback severity of 50.0 applied)*")

    # ── PROJECT INFORMATION & DISCLAIMER ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 Project System Architecture & Disclaimer")

    # Prominent Academic / Research Disclaimer
    st.warning(
        "⚠️ **ACADEMIC RESEARCH & DEMO DISCLAIMER:**\n\n"
        "This application is a coursework research prototype developed for the Soft Computing project (SC08) "
        "using the **UCI AI4I 2020 Predictive Maintenance Dataset**.\n"
        "- It is **NOT** a certified industrial safety instrumented system (SIS) or SIL-rated controller.\n"
        "- The AI4I 2020 dataset does **not** contain vibration sensors; spindle load is represented by **Torque [Nm]**.\n"
        "- Operating decisions in commercial facilities must always be validated by qualified plant maintenance engineers and certified diagnostic equipment."
    )

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.expander("🔄 Complete Data Processing Pipeline", expanded=False):
            st.markdown("""
            <div style="display: flex; flex-direction: column; gap: 4px; padding: 4px 0;">
                <div class="flow-step-box">1. Raw Sensor Input (6 parameters)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">2. Validation & Min-Max Normalisation (norm_params.json)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">3. Hand-coded ANN (7 → 16 → 8 → 1, ann_weights.npz)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">4. Continuous Fault Probability P(Failure) ∈ [0, 1]</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">5. Mamdani Fuzzy Logic System (11 IF–THEN rules)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">6. CoG Centroid Defuzzification → Severity Index (0–100)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">7. Priority Assignment: LOW / MEDIUM / HIGH / URGENT</div>
            </div>
            """, unsafe_allow_html=True)

    with exp_col2:
        with st.expander("ℹ️ Model & Universe Specifications", expanded=False):
            st.markdown("""
            - **ANN Architecture:** 7 Inputs → 16 Hidden (ReLU) → 8 Hidden (ReLU) → 1 Output (Sigmoid)
            - **Weights Source:** Pre-trained weights loaded from `results/ann_weights.npz`
            - **Classification Threshold:** $\\tau = 0.50$ (evaluated independently from fuzzy severity)
            - **Fuzzy Universe Limits:**
              - `fault_probability`: $[0.0, 1.0]$
              - `tool_wear_min`: $[0.0, 260.0]$ minutes
              - `torque_Nm`: $[0.0, 80.0]$ Nm *(Torque, NOT vibration)*
            - **Fuzzy Priority Thresholds:**
              - Severity $\\ge 78.0$: `URGENT`
              - $60.0 \\le$ Severity $< 78.0$: `HIGH`
              - $33.0 \\le$ Severity $< 60.0$: `MEDIUM`
              - Severity $< 33.0$: `LOW`
            """)

    # Footer
    st.markdown("""
        <div class="footer-text">
            SC08 | Machine Predictive Maintenance & Fault Priority System &bull; Soft Computing Coursework Product V1
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

