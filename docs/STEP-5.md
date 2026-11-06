# Step 5 — Product V1 System Documentation

> **Academic Milestone:** Step 5 — Product V1 (Local Streamlit Decision Support System)  
> **Course / Project Code:** SC08 — Soft Computing  
> **Project Title:** Machine Predictive Maintenance & Fault Priority System  
> **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)  

---

## 1. Project Information

- **Project Title:** SC08 — Machine Predictive Maintenance & Fault Priority System
- **Course / Track:** SC08 — Soft Computing
- **Repository Remote:** `https://github.com/Anuj-191/Machine-Predictive-Maintenance-system.git`
- **Application Version:** Product V1 (Interactive Local Streamlit Decision Support Dashboard)
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (UCI Repository ID: 601)

---

## 2. Product V1 Objective and User Task

The primary objective of **Product V1** is to provide an accessible, interactive, transparent decision-support system that translates complex soft-computing models into actionable industrial maintenance guidance.

### Target User Task
A factory operator, reliability engineer, or coursework examiner uses this dashboard to:
1. Input current operating conditions from milling machine sensors (or load standard scenario presets).
2. Obtain an immediate, data-driven estimate of component breakdown risk from a hand-coded Artificial Neural Network (ANN).
3. Evaluate physical machine degradation (tool wear and spindle torque) through a hand-coded Mamdani Fuzzy Inference System.
4. Review an interpretable **Severity Index ($0\text{--}100$)**, a discrete **Maintenance Priority** (`LOW`, `MEDIUM`, `HIGH`, `URGENT`), and specific action recommendations.
5. Inspect the transparent rule activations and firing strengths that explain *why* the hybrid system reached its recommendation.

---

## 3. Application Interface Type

- **Interface Framework:** Streamlit Local Web Application (`streamlit>=1.35,<2.0`).
- **Execution Architecture:** Pure Python client-server architecture running locally on `http://localhost:8501`.
- **Decoupled Architecture:** The web interface serves strictly as an I/O presentation layer. It performs no internal model training, feature normalization, or inference duplication; all calculations are executed by delegating to the unified backend module `src/prediction.py`.

---

## 4. Official Entry Point

- **Canonical Application Entry Point:** [`app.py`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/app.py) (located at the repository root).
- **Subfolder Alias:** [`app/app.py`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/app/app.py) is maintained as a synchronized alias for backward compatibility. Running either file executes the identical interface.

---

## 5. Exact Launch Command

To launch the Product V1 application from the project root in any terminal:

```bash
streamlit run app.py
```

*(Alternatively, `streamlit run app/app.py`)*

---

## 6. Saved ANN Model Path

- **Artifact File:** [`results/ann_weights.npz`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/results/ann_weights.npz)
- **Format:** Uncompressed NumPy array archive (`.npz`).
- **Contained Arrays & Dimensions:**
  - `W1`: Layer 1 weights $\in \mathbb{R}^{7 \times 16}$
  - `b1`: Layer 1 bias $\in \mathbb{R}^{1 \times 16}$
  - `W2`: Layer 2 weights $\in \mathbb{R}^{16 \times 8}$
  - `b2`: Layer 2 bias $\in \mathbb{R}^{1 \times 8}$
  - `W3`: Layer 3 weights $\in \mathbb{R}^{8 \times 1}$
  - `b3`: Layer 3 bias $\in \mathbb{R}^{1 \times 1}$
  - **Total Trainable Parameters:** $112 + 16 + 128 + 8 + 8 + 1 = 273$ parameters.
- **Model Training Record:** Trained using mini-batch SGD with positive class weighting ($w_{\text{pos}} = 28.5203$) and early stopping on validation loss (optimal model restored from epoch 71).

---

## 7. Normalization Parameter Path

- **Artifact File:** [`data/processed/norm_params.json`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/data/processed/norm_params.json)
- **Scope:** Computed strictly from the 8,000-sample training partition (`X_train.npy`) to prevent data leakage.
- **Scaling Parameters Table:**

| Feature Name | Training Min | Training Max | Scaling Formula |
| :--- | :---: | :---: | :--- |
| **Air temperature [K]** | $295.3$ | $304.5$ | $(T_{\text{air}} - 295.3) / (304.5 - 295.3)$ |
| **Process temperature [K]** | $305.7$ | $313.8$ | $(T_{\text{proc}} - 305.7) / (313.8 - 305.7)$ |
| **Rotational speed [rpm]** | $1168.0$ | $2886.0$ | $(\omega - 1168.0) / (2886.0 - 1168.0)$ |
| **Torque [Nm]** | $3.8$ | $76.6$ | $(\tau - 3.8) / (76.6 - 3.8)$ |
| **Tool wear [min]** | $0.0$ | $253.0$ | $(t_{\text{wear}} - 0.0) / (253.0 - 0.0)$ |

---

## 8. Six Raw Input Parameters, Types, and Units

The application collects exactly six raw operating inputs from the user interface:

| Parameter Label | Variable Name | Data Type | Physical Units | UI Input Range | Description / Physical Meaning |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Product Type** | `product_type` | `str` | Category | `{'L', 'M', 'H'}` | Machine variant quality: Low (50%), Medium (30%), High (20%). |
| **Air Temperature** | `air_temp_K` | `float` | Kelvin [K] | $[250.0, 350.0]$ | Surrounding ambient air intake temperature. |
| **Process Temperature** | `process_temp_K` | `float` | Kelvin [K] | $[250.0, 350.0]$ | Cutting zone operational temperature. |
| **Rotational Speed** | `rotational_speed_rpm` | `float` | rpm | $[0.0, 5000.0]$ | Spindle angular rotational speed. |
| **Torque** | `torque_Nm` | `float` | Newton-metres [Nm] | $[0.0, 80.0]$ | Spindle load torque. *(Capped at 80 Nm to match fuzzy universe)* |
| **Tool Wear** | `tool_wear_min` | `float` | Minutes [min] | $[0.0, 260.0]$ | Cumulative tool cutting time. *(Capped at 260 min to match fuzzy universe)* |

---

## 9. Seven ANN Input Features and Canonical Order

To maintain mathematical compatibility with the pre-trained weight matrix $W_1 \in \mathbb{R}^{7 \times 16}$, inputs are preprocessed into a 7-element vector in this exact order:

```
Index 0: Air temperature [K]        (Min-Max scaled continuous float)
Index 1: Process temperature [K]    (Min-Max scaled continuous float)
Index 2: Rotational speed [rpm]     (Min-Max scaled continuous float)
Index 3: Torque [Nm]                (Min-Max scaled continuous float)
Index 4: Tool wear [min]            (Min-Max scaled continuous float)
Index 5: Type_M                     (One-hot binary flag: 1.0 if Type=='M', else 0.0)
Index 6: Type_H                     (One-hot binary flag: 1.0 if Type=='H', else 0.0)
```

*(Note: Type L is the implicit base category when `Type_M = 0.0` and `Type_H = 0.0`)*

---

## 10. Output Fields, Units, Meaning, and Priority Levels

Executing a prediction produces a structured output dictionary containing seven fields:

| Field Key | Data Type | Range / Domain | Meaning & Operational Significance |
| :--- | :---: | :---: | :--- |
| `fault_probability` | `float` | $[0.0, 1.0]$ | ANN output probability $P(\text{Failure} \mid X)$ rounded to 4 decimals. |
| `severity_index` | `float` | $[0.0, 100.0]$ | Defuzzified continuous severity index from Mamdani CoG integration. |
| `priority` | `str` | 4 Levels | Maintenance urgency tier: `LOW`, `MEDIUM`, `HIGH`, or `URGENT`. |
| `recommendation` | `str` | Text | Human-readable actionable operational instruction. |
| `fuzzy_explanation` | `str` | Text | Transparent synthesis explaining how inputs produced the decision. |
| `fired_rules` | `list[dict]` | Variable | Detailed records of all active rules with firing strengths $> 0.0$. |
| `raw_inputs` | `dict` | Key-value | Type-cast validated input readings confirming recorded telemetry. |

### Priority Assignment Matrix

| Priority Level | Severity Score Range | Banner / Badge Color | Operational Action |
| :---: | :---: | :---: | :--- |
| **LOW** | $\text{Severity} < 33.0$ | **Green** (`#16A34A`) | Routine monitoring only. Continue normal machine operation. |
| **MEDIUM** | $33.0 \le \text{Severity} < 60.0$ | **Yellow** (`#EAB308`) | Plan tool inspection and lubrication within next 2–3 operating days. |
| **HIGH** | $60.0 \le \text{Severity} < 78.0$ | **Orange** (`#EA580C`) | Schedule maintenance within 24 hours. Reduce spindle load. |
| **URGENT** | $\text{Severity} \ge 78.0$ | **Red** (`#DC2626`) | **STOP MACHINE IMMEDIATELY.** Perform full mechanical inspection. |

---

## 11. End-to-End Data Flow

The sequential data flow from user interaction to final rendered assessment proceeds through six deterministic steps:

```
[ Factory Operator / Browser UI ]
                 │
                 ▼  (Raw values: product_type, air_temp, proc_temp, speed, torque, tool_wear)
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Defensive Input Validation (`validate_inputs` in src/prediction.py) │
│    - Checks types (TypeError) and physical boundaries (ValueError)     │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Feature Preprocessing (`_encode_and_normalize` in src/prediction.py)│
│    - One-hot encodes Type into Type_M and Type_H                       │
│    - Normalises 5 numeric sensors via norm_params.json                 │
│    - Forms canonical 1×7 float64 vector                                │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. ANN Forward Pass (`ann_model.forward` in src/ann.py)                │
│    - Layer 1: z1 = X @ W1 + b1  -> a1 = ReLU(z1)      [7 -> 16]        │
│    - Layer 2: z2 = a1 @ W2 + b2 -> a2 = ReLU(z2)      [16 -> 8]        │
│    - Layer 3: z3 = a2 @ W3 + b3 -> y_hat = Sigmoid(z3)[8 -> 1]         │
│    - Extract fault_probability = float(y_hat[0, 0]) ∈ [0, 1]           │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Mamdani Fuzzy Logic Inference (`fuzzy_system.evaluate` in src/fuzzy)│
│    - Fuzzification: 3 inputs (fault_prob, tool_wear, torque)           │
│    - Rule Evaluation: 11 IF-THEN rules via Gödel min-conjunction       │
│    - Mamdani Implication: Consequent MFs clipped at rule strength      │
│    - Aggregation: Pointwise maximum (S-norm) over 1000 discrete points │
│    - Defuzzification: Centre of Gravity (CoG) -> Severity Index [0-100]│
│    - Priority Assignment: Maps severity to LOW / MEDIUM / HIGH / URGENT│
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Decision Packaging (`PredictiveMaintenancePipeline.predict`)        │
│    - Assembles synthesized explanation and fired rules list            │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. Streamlit Presentation Rendering (`app.py`)                         │
│    - Displays ANN binary threshold decision (P >= 0.50 warning)        │
│    - Displays metric cards, severity progress gauge, priority badge    │
│    - Renders action box and expandable explainability accordion        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Python Version and Repository Dependencies

### Python Environment
- **Required Version:** Python $\ge 3.9$
- **Tested Environment:** Python `3.12.9` (64-bit, Windows)

### Pinned Dependencies ([`requirements.txt`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/requirements.txt))

| Package | Version Constraint | Active Role in SC08 System |
| :--- | :--- | :--- |
| `numpy` | `>=1.24,<2.0` | Core computational array library: ANN backprop/forward pass, matrix math, fuzzy logic CoG Riemann integration. |
| `pandas` | `>=2.0,<3.0` | Tabular data loading and dataset preprocessing pipelines. |
| `scikit-learn` | `>=1.3,<2.0` | Evaluation metrics strictly (`confusion_matrix`, `roc_auc_score`). *Not used for model building.* |
| `matplotlib` | `>=3.7,<4.0` | Evaluation charting, learning curves, and offline figure exports. |
| `streamlit` | `>=1.35,<2.0` | Web application framework powering the interactive Product V1 dashboard. |
| `joblib` | `>=1.3,<2.0` | Parallel computation utility for evaluation benchmarks. |

*Explicitly Excluded Libraries: TensorFlow, PyTorch, Keras, scikit-fuzzy (`skfuzzy`), and `imbalanced-learn` (all ANN and Fuzzy modules are written from scratch using pure NumPy).*

---

## 13. Installation and Launch Instructions

### Prerequisites
Ensure Git and Python $\ge 3.9$ are installed on the local system.

### Step 1: Clone Repository and Navigate to Root
```bash
git clone https://github.com/Anuj-191/Machine-Predictive-Maintenance-system.git
cd Machine-Predictive-Maintenance-system
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Weights and Parameters
Ensure pre-trained weights and normalization parameters exist:
```bash
python -c "import os; assert os.path.exists('results/ann_weights.npz'); assert os.path.exists('data/processed/norm_params.json'); print('Assets verified.')"
```

### Step 5: Launch Product V1 Dashboard
```bash
streamlit run app.py
```
Open a browser and navigate to `http://localhost:8501`.

---

## 14. Actual Test Cases and Empirical Verification Results

### A. Operational Boundary Test Cases

| Test Case Name | Evaluated Inputs | ANN Prob $P$ | Severity Index | Priority | Fired Rules & Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Normal Input** *(Preset Scenario 1)* | Type: `L`<br>$T_{\text{air}} = 300\,\text{K}, T_{\text{proc}} = 310\,\text{K}$<br>$\omega = 1500\,\text{rpm}, \tau = 40\,\text{Nm}, t_{\text{wear}} = 20\,\text{min}$ | $0.0216$ ($2.16\%$) | $15.53 / 100$ | **LOW** | R1 fired at strength $1.0000$. Machine healthy, routine monitoring. |
| **Minimum Boundary** *(Lower physical limit)* | Type: `L`<br>$T_{\text{air}} = 250\,\text{K}, T_{\text{proc}} = 250\,\text{K}$<br>$\omega = 0\,\text{rpm}, \tau = 0\,\text{Nm}, t_{\text{wear}} = 0\,\text{min}$ | $0.0253$ ($2.53\%$) | $15.53 / 100$ | **LOW** | R1 fired at strength $1.0000$. Validated safe response at physical zero. |
| **Emergency Scenario** *(Preset Scenario 5)* | Type: `M`<br>$T_{\text{air}} = 304\,\text{K}, T_{\text{proc}} = 314\,\text{K}$<br>$\omega = 1180\,\text{rpm}, \tau = 68\,\text{Nm}, t_{\text{wear}} = 235\,\text{min}$ | $0.9982$ ($99.82\%$) | $84.47 / 100$ | **URGENT** | R9 ($1.0000$) and R10 ($1.0000$) fired. Emergency shutdown required. |
| **Maximum Supported UI Boundary** | Type: `M`<br>$T_{\text{air}} = 304\,\text{K}, T_{\text{proc}} = 314\,\text{K}$<br>$\omega = 1200\,\text{rpm}, \tau = 80\,\text{Nm}, t_{\text{wear}} = 260\,\text{min}$ | $0.9999$ ($99.99\%$) | $84.47 / 100$ | **URGENT** | Full saturation of fuzzy high-wear and high-torque universes. |
| **Theoretical Max Limit** *(All bounds max)* | Type: `H`<br>$T_{\text{air}} = 350\,\text{K}, T_{\text{proc}} = 350\,\text{K}$<br>$\omega = 5000\,\text{rpm}, \tau = 80\,\text{Nm}, t_{\text{wear}} = 260\,\text{min}$ | $1.0000$ ($100.0\%$) | $84.47 / 100$ | **URGENT** | R9 and R10 fully active. Maximum severity ceiling reached. |

---

### B. Specific Boundary Rejections & Direct Pipeline Validation

1. **Torque Above 80 Nm UI Prevention:**
   - Setting $\tau > 80.0\,\text{Nm}$ in the browser is blocked by widget parameter `max_value=80.0`.
   - Programmatic bypass attempts trigger the Python guard in `app.py`:
     `Input Validation Error: Torque value (85.0 Nm) exceeds the maximum supported fuzzy universe limit of 80.0 Nm.`
2. **Tool Wear Above 260 min UI Prevention:**
   - Setting $t_{\text{wear}} > 260.0\,\text{min}$ in the browser is blocked by widget parameter `max_value=260.0`.
   - Programmatic bypass attempts trigger the Python guard in `app.py`:
     `Input Validation Error: Tool wear value (275.0 min) exceeds the maximum supported fuzzy universe limit of 260.0 min.`
3. **Invalid Values Tested Directly Against `src/prediction.py`:**
   - `product_type = None` $\to$ Rejection: `TypeError: product_type must be a string, got NoneType.`
   - `product_type = ""` $\to$ Rejection: `ValueError: Invalid product_type ''. Must be one of: ['H', 'L', 'M'].`
   - `air_temp_K = None` $\to$ Rejection: `TypeError: Input 'air_temp_K' must be numeric (int/float), got NoneType.`
   - `air_temp_K = ""` $\to$ Rejection: `TypeError: Input 'air_temp_K' must be numeric (int/float), got str.`
   - `air_temp_K = float('nan')` $\to$ Rejection: `ValueError: Input 'air_temp_K' value nan is out of valid physical range [250.0, 350.0].`
   - `air_temp_K = float('inf')` $\to$ Rejection: `ValueError: Input 'air_temp_K' value inf is out of valid physical range [250.0, 350.0].`
   - `air_temp_K = float('-inf')` $\to$ Rejection: `ValueError: Input 'air_temp_K' value -inf is out of valid physical range [250.0, 350.0].`
4. **Missing Model Weights & Parameters Behavior:**
   - If `results/ann_weights.npz` is missing, `PredictiveMaintenancePipeline` raises an explicit `FileNotFoundError`:
     `ANN weights file not found at 'results/ann_weights.npz'. Run 'python -m src.ann' first to train and save weights.`
   - If `data/processed/norm_params.json` is missing, it raises:
     `Normalisation parameters file not found at 'data/processed/norm_params.json'. Run 'python -m src.preprocessing' first.`
   - In `app.py`, this is caught by `try...except Exception as e`, displaying an error callout to the user and halting execution via `st.stop()`.
5. **UI versus Direct Pipeline Consistency:**
   - Direct execution via Python and UI execution via browser form submission yield identical floating-point values down to 4 decimal places for probability ($0.0216$) and 2 decimal places for severity ($15.53$).

---

### C. Detailed Probability Discrepancy Reconciliation

During project verification, two high-risk probability figures were recorded:
- **Case 1:** $99.82\%$ in the screenshot report (`03-urgent-prediction.png`).
- **Case 2:** $99.99\%$ in the maximum-boundary verification script.

**Mathematical Explanation & Reproduction:**
1. **Case 1 ($99.82\%$)** reproduces the exact inputs configured in the dashboard's **"Scenario 5: Critical Emergency Condition"** preset:
   - Inputs: Type `M`, $T_{\text{air}} = 304.0\,\text{K}$, $T_{\text{proc}} = 314.0\,\text{K}$, $\omega = 1180.0\,\text{rpm}$, $\tau = 68.0\,\text{Nm}$, $t_{\text{wear}} = 235.0\,\text{min}$.
   - ANN forward pass produces $\hat{y} = 0.9982052\dots$, which formats as **$99.82\%$** (Prob: $0.9982$).
2. **Case 2 ($99.99\%$)** represents the **extreme ceiling test** where physical parameters were pushed to the absolute universe limits:
   - Inputs: Type `M`, $T_{\text{air}} = 304.0\,\text{K}$, $T_{\text{proc}} = 314.0\,\text{K}$, $\omega = 1200.0\,\text{rpm}$, $\tau = 80.0\,\text{Nm}$ (universe max), $t_{\text{wear}} = 260.0\,\text{min}$ (universe max).
   - ANN forward pass produces $\hat{y} = 0.9999201\dots$, which formats as **$99.99\%$** (Prob: $0.9999$).
3. **Conclusion:** Both cases represent severe physical degradation activating Rules R9 and R10, yielding the same maximal Severity Index ($84.47 / 100$) and assigning `URGENT` priority. The numeric probability difference reflects the elevated torque ($80\,\text{Nm}$ vs $68\,\text{Nm}$) and tool wear ($260\,\text{min}$ vs $235\,\text{min}$) in the boundary test.

---

## 15. Product V1 Screenshot Evidence

All evidence screenshots are stored under [`docs/evidence/step5/`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/):

| Screenshot Reference | Relative File Path | Verified Visual Content |
| :--- | :--- | :--- |
| **Figure 1: Input Controls** | [`docs/evidence/step5/01-input-form.png`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/01-input-form.png) | Complete sidebar showing preset loader, universe boundary notice ($\le 80\,\text{Nm}, \le 260\,\text{min}$), input controls, and Run button. |
| **Figure 2: Normal Operation** | [`docs/evidence/step5/02-normal-prediction.png`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/02-normal-prediction.png) | Normal machine state (Scenario 1): Healthy ANN banner ($2.16\%$), Severity $15.53 / 100$, solid Green `LOW` badge. |
| **Figure 3: Urgent Condition** | [`docs/evidence/step5/03-urgent-prediction.png`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/03-urgent-prediction.png) | Emergency state (Scenario 5): Failure Predicted banner ($99.82\%$), Severity $84.47 / 100$, solid Red `URGENT` badge. |
| **Figure 4: Explainability & Rules** | [`docs/evidence/step5/04-explanation-and-rules.png`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/04-explanation-and-rules.png) | Action recommendation, synthesis explanation, and fired fuzzy rule accordion detailing active rules (R9 & R10). |
| **Figure 5: Dark Mode Theme** | [`docs/evidence/step5/05-dark-mode.png`](file:///d:/COURSES/Project/Machine-Predictive-Maintenance-system/docs/evidence/step5/05-dark-mode.png) | Full dashboard in Streamlit Dark Mode, demonstrating theme-adaptive heading contrast, card backgrounds, and badges. |

---

## 16. Earlier Feedback Addressed

### A. Critical Physical Distinction: Torque vs. Vibration
- **The Issue:** Predictive maintenance literature frequently references vibration sensors. Previous project drafts required clarification regarding spindle sensors.
- **Correction Implemented:** The UCI AI4I 2020 dataset does **not** contain an accelerometer, vibration transducer, or vibration signal. Spindle load is recorded strictly as **Torque in Newton-metres [Nm]**.
- **System Fidelity:** Across all source files (`src/preprocessing.py`, `src/prediction.py`, `src/fuzzy.py`, `app.py`, and documentation), `Torque [Nm]` is rigorously documented and labeled as torque. It is never treated, renamed, or proxied as vibration.

### B. Independent ANN Decision vs. Fuzzy Priority
- In initial prototypes, operators could confuse the ANN probability classification with the fuzzy priority label.
- In Product V1, an explicit **ANN Threshold Decision Banner** is rendered at the top of results. It clearly flags whether $P(\text{Failure}) \ge 0.50$ (Failure Predicted) or $< 0.50$ (Healthy) based strictly on the validated neural classification threshold ($\tau = 0.50$), presented distinctly from the downstream fuzzy Severity Index and priority tier.

### C. UI Input Constraint Alignment
- The fuzzy logic inference engine defines membership universes up to $260\,\text{min}$ for tool wear and $80\,\text{Nm}$ for torque.
- Product V1 constrains the UI number inputs to these limits with an informative sidebar notice, preventing users from submitting unmapped out-of-universe values.

---

## 17. Limitations and Responsible-Use Disclaimer

1. **Synthetic Nature of AI4I 2020 Dataset:** The AI4I 2020 dataset was generated synthetically using physical simulation models (Matzka, 2020). While it models realistic thermodynamic and mechanical failure modes, empirical performance on diverse CNC machinery from different machine vendors may vary.
2. **Academic Research Scope:** This application is an educational prototype developed for the Soft Computing course (SC08). It is **not** certified under industrial functional safety standards (such as IEC 61508, ISO 13849, or SIL ratings).
3. **Safety Interlock Restriction:** The system is designed solely as an operator decision-support advisory tool. It must **not** be integrated directly into physical machine emergency stop (E-stop) circuits or safety PLC loops without hardware redundancy and physical validation by certified plant maintenance engineers.



