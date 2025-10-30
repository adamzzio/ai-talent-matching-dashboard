# AI Talent Matching & Dashboard

An AI-powered analytics application that transforms SQL-based talent data into actionable insights through a **parameterized, dynamic dashboard**. This app integrates **PostgreSQL/Supabase**, **Python**, and **Streamlit** to create an interactive interface for evaluating employee–job fit in real time.

**Author : Adam Maurizio Winata**
**LinkedIn** : [LinkedIn](https://www.linkedin.com/in/adammauriziowinata/)

---

## 🧭 Overview

Your final step is to turn the SQL results into actionable insight through a **parameterized, AI-powered interface and visual dashboard**. This app dynamically handles any new job input at runtime — not a static or pre-baked dashboard.

**Live App Deployment Link** : [AI Talent App & Dashboard](https://ai-talent-matching-dashboard.streamlit.app/)

### 🎯 Core Functions

**Ranked Talent List**

* Displays the output of your SQL logic including: `employee_id`, `name`, `final_match_rate`, and supporting fields (e.g., top TVs/TGVs, strengths, gaps)

**Dashboard Visualization**

* Provides clear, interactive visuals for each new input/job vacancy:

  * Match-rate distributions
  * Benchmark vs candidate comparisons (radar, heatmap, bar plots)
  * Summary insights explaining why certain employees rank highest

✅ The dashboard helps stakeholders easily **see, understand, and act on** the data.

---

## ✨ Features

* **Streamlit UI** for interactive and parameterized dashboards
* **PostgreSQL/Supabase** backend for scalable data management
* **Hybrid computation** — SQL for retrieval, Python for AI-based logic and scoring
* **AI Talent Job Description Generator** — automatically generates structured job descriptions using AI input
* **Employee Match Rate (TV–TGV) Calculation** — compute match rates between employee traits and vacancy benchmarks
* **Dynamic visualization** — radar, bar, and heatmap charts for TGV/TV comparisons
* **Report exports** — generate PDF and Excel outputs


---

## 🧱 Tech Stack

* Python 3.13.5
* Streamlit
* pandas / numpy
* psycopg2, supabase-py
* ReportLab (PDF)
* Groq SDKs for AI-powered features

---

## 📂 Project Structure

```
├─ sql/
│  ├─ cte_employees.sql       # CTE to get All Employee Data
│  ├─ cte_benchmark.sql       # CTE to get Benchmark Data
├─ .streamlit/                # Folder for secrets.toml
│  ├─ secrets.toml            # Stored in Streamlit Cloud (not pushed to GitHub)
├─ .env                       # Store environment variables
├─ access_supabase.py         # Test connection to Supabase
├─ export_csv.py              # Load XLSX sheets, convert to CSV, and import to Supabase
├─ app.py                     # Main Streamlit dashboard app
├─ requirements.txt           # List of dependencies
├─ solution_step_1.ipynb      # Notebook solution for challenge step 1
├─ solution_step_2.ipynb      # Notebook solution for challenge step 2
└─ README.md
```

---

## 🧩 Configuration

### `.streamlit/secrets.toml`

Use this to store database credentials safely.

```toml
[supabase]
host = "YOUR_DB_HOST"
port = "5432"
database = "YOUR_DB_NAME"
user = "YOUR_DB_USER"
password = "YOUR_DB_PASSWORD"
pool_mode = "transaction"

[groq]
GROQ_API_KEY = "SUPABASE_API_KEY"
GROQ_BASE_URL = "SUPABASE_BASE_URL"
MODEL = "MODEL_NAME"
```

### `.env`

Used for SDKs and other environment variables.

```bash
host="YOUR_DB_HOST"
port="5432"
database="YOUR_DB_NAME"
user="YOUR_DB_USER"
password="YOUR_DB_PASSWORD"
pool_mode="transaction"

GROQ_API_KEY="SUPABASE_API_KEY"
GROQ_BASE_URL="SUPABASE_BASE_URL"
MODEL="MODEL_NAME"
```

---

## 🚀 Getting Started

### 🧩 Prerequisites

* Python 3.13.5
* VSCode
* Supabase project (with access credentials)

### 🧰 Installation

```bash
# 1) Clone repository
$ git clone https://github.com/adamzzio/ai-talent-matching-dashboard.git

# 2) Open in VSCode
# 3) Install dependencies
$ pip install -r requirements.txt

# 4) Run Streamlit app
$ streamlit run app.py
```

Open the local URL displayed by Streamlit (usually `http://localhost:8501`).

---

## 📊 Example Outputs

* **Ranked Talent Table** with `final_match_rate`, strengths, etc.
* **Radar Chart**: Benchmark vs Candidate TGV profiles
* **Distribution Plots**: Match rate across employees
* **Narrative Summary**: AI-generated explanation for top-ranked candidates

---

## 🧠 References

* [Streamlit Docs](https://docs.streamlit.io/)
* [Groq API Docs](https://console.groq.com/docs/overview)
* [Supabase Docs](https://supabase.com/docs)
* [ChatGPT](https://chatgpt.com/)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR for major changes before submission.