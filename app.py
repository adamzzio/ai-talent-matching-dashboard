# ==========================
# ===== Import Library =====
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import re
import plotly.graph_objects as go

import streamlit as st
import os
import supabase
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

from openai import OpenAI
import json
import textwrap

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY

# ==========================
# ===== Load ENV =====
# ==========================
HOST = st.secrets['supabase']['host']
PORT = st.secrets['supabase']['port']
DBNAME = st.secrets['supabase']['database']
USER = st.secrets['supabase']['user']
POOL = st.secrets['supabase']['pool_mode']
PASSWORD = st.secrets['supabase']['password']

API_KEY = st.secrets['groq']['GROQ_API_KEY']
BASE_URL = st.secrets['groq']['GROQ_BASE_URL']
MODEL = st.secrets['groq']['MODEL']

# ==========================
# ===== Create Function to Get Employee Benchmark Data =====
# ==========================
def employees_benchmark_data(employee_ids, schema: str = "public") -> pd.DataFrame:
    """
    Build a consolidated employee benchmark dataset by joining:
      - Employees master (IDs and dimension FKs)
      - Latest competencies score per employee (yearly history, take the newest year)
      - Psychological profiles (pauli, DISC word, MBTI, IQ, GTQ, TIKI)
      - Strengths (top-ranked theme per employee)
      - PAPI scores (scale and score)
      - Dimension tables to resolve human-readable names:
          * directorate name    (dim_directorates.name)
          * role/position name  (dim_positions.name)
          * grade name          (dim_grades.name)

    Output columns:
      employee_id, full_name, directorate, role, grade, score, pauli,
      disc_1, disc_2, mbti_1, mbti_2, mbti_3, mbti_4,
      iq, gtq, tiki, strengths, scale_code, papi_scores
    """
    if not employee_ids:
        return pd.DataFrame(columns=[
            "employee_id","full_name","directorate","role","grade","score",
            "pauli","disc_1","disc_2","mbti_1","mbti_2","mbti_3","mbti_4",
            "iq","gtq","tiki","strengths","scale_code","papi_scores"
        ])

    assert schema.replace("_", "").isalnum(), "Invalid schema name"

    qry = f"""
    WITH
    ids AS (
      SELECT UNNEST(%s::text[]) AS employee_id
    ),
    employees_base AS (
      SELECT
        e.employee_id,
        e.fullname AS full_name,
        e.nip,
        e.directorate_id,
        e.position_id,
        e.grade_id
      FROM {schema}.employees e
      JOIN ids i ON i.employee_id = e.employee_id::text
    ),
    competencies_latest AS (
      SELECT DISTINCT ON (cy.employee_id)
             cy.employee_id,
             cy.score,
             cy.year
      FROM {schema}.competencies_yearly cy
      JOIN ids i ON i.employee_id = cy.employee_id::text
      WHERE cy.year IS NOT NULL
      ORDER BY cy.employee_id, cy.year DESC
    ),
    profiles_psych_f AS (
      SELECT
        p.employee_id,
        p.pauli,
        p.disc_word,
        p.mbti,
        p.iq,
        p.gtq,
        p.tiki
      FROM {schema}.profiles_psych p
      JOIN ids i ON i.employee_id = p.employee_id::text
    ),
    strengths_rank AS (
      SELECT
        s.employee_id,
        s.rank,
        s.theme,
        ROW_NUMBER() OVER (PARTITION BY s.employee_id ORDER BY s.rank ASC, s.theme) AS rn
      FROM {schema}.strengths s
      JOIN ids i ON i.employee_id = s.employee_id::text
    ),
    strengths_top AS (
      SELECT employee_id, theme
      FROM strengths_rank
      WHERE rn = 1
    ),
    papi_f AS (
      SELECT
        ps.employee_id,
        ps.scale_code,
        ps.score AS papi_scores
      FROM {schema}.papi_scores ps
      JOIN ids i ON i.employee_id = ps.employee_id::text
    ),
    -- Dimension CTEs: map IDs to display names
    dim_directorates_f AS (
      SELECT directorate_id, name AS directorate
      FROM {schema}.dim_directorates
    ),
    dim_positions_f AS (
      SELECT position_id, name AS role
      FROM {schema}.dim_positions
    ),
    dim_grades_f AS (
      SELECT grade_id, name AS grade
      FROM {schema}.dim_grades
    ),
    emp_with_profiles AS (
      SELECT
        eb.employee_id,
        eb.full_name,
        eb.nip,
        eb.directorate_id,
        eb.position_id,
        eb.grade_id,
        pp.pauli,
        pp.disc_word,
        pp.mbti,
        pp.iq,
        pp.gtq,
        pp.tiki
      FROM employees_base eb
      LEFT JOIN profiles_psych_f pp USING (employee_id)
    ),
    emp_with_strengths AS (
      SELECT
        ewp.*,
        st.theme AS strengths
      FROM emp_with_profiles ewp
      LEFT JOIN strengths_top st USING (employee_id)
    ),
    emp_with_papi AS (
      SELECT
        ews.*,
        pf.scale_code,
        pf.papi_scores
      FROM emp_with_strengths ews
      LEFT JOIN papi_f pf USING (employee_id)
    ),
    -- Join human-readable names from dimensions
    emp_with_names AS (
      SELECT
        ewp.*,
        dd.directorate,
        dp.role,
        dg.grade
      FROM emp_with_papi ewp
      LEFT JOIN dim_directorates_f dd ON dd.directorate_id = ewp.directorate_id
      LEFT JOIN dim_positions_f    dp ON dp.position_id    = ewp.position_id
      LEFT JOIN dim_grades_f       dg ON dg.grade_id       = ewp.grade_id
    ),
    -- Parse MBTI/DISC into separate columns
    parsed_splits AS (
      SELECT
        ewp.employee_id,
        -- DISC split
        NULLIF(split_part(ewp.disc_word, '-', 1), '') AS disc_1,
        NULLIF(split_part(ewp.disc_word, '-', 2), '') AS disc_2,

        -- MBTI parsed 4 kolom
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'I' THEN 'Introversion'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'E' THEN 'Extraversion'
          ELSE NULL
        END AS mbti_1,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'S' THEN 'Sensing'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'N' THEN 'Intuition'
          ELSE NULL
        END AS mbti_2,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'T' THEN 'Thinking'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'F' THEN 'Feeling'
          ELSE NULL
        END AS mbti_3,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'J' THEN 'Judging'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'P' THEN 'Perceiving'
          ELSE NULL
        END AS mbti_4
      FROM emp_with_names ewp
    ),
    -- Assemble raw output (may have duplicates)
    final_raw AS (
      SELECT
        ewp.employee_id,
        ewp.full_name,
        ewp.directorate,   -- nama direktorat
        ewp.role,          -- nama role/position
        ewp.grade,         -- nama grade
        COALESCE(cl.score, 0)  AS score,
        COALESCE(ewp.pauli, 0) AS pauli,
        ps.disc_1,
        ps.disc_2,
        ps.mbti_1,
        ps.mbti_2,
        ps.mbti_3,
        ps.mbti_4,
        COALESCE(ewp.iq,  0)   AS iq,
        COALESCE(ewp.gtq, 0)   AS gtq,
        COALESCE(ewp.tiki,0)   AS tiki,
        ewp.strengths          AS strengths,
        ewp.scale_code,
        COALESCE(ewp.papi_scores, 0) AS papi_scores
      FROM emp_with_names ewp
      LEFT JOIN competencies_latest cl USING (employee_id)
      LEFT JOIN parsed_splits     ps USING (employee_id)
    )
    -- Final select with de-duplication
    SELECT DISTINCT
      employee_id,
      full_name,
      directorate,
      role,
      grade,
      score,
      pauli,
      disc_1,
      disc_2,
      mbti_1,
      mbti_2,
      mbti_3,
      mbti_4,
      iq,
      gtq,
      tiki,
      strengths,
      scale_code,
      papi_scores
    FROM final_raw
    ORDER BY employee_id, full_name;
    """

    # ===== Connection to Supabase =====
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD,
        sslmode="require",
        connect_timeout=30,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(qry, (employee_ids,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()

# ==========================
# ===== Create Function to Get Employee Data =====    
# ==========================
def get_all_employee_data(schema: str = "public") -> pd.DataFrame:
    """
    Retrieve ALL employees with:
      - latest competencies score per employee,
      - psychological profiles (pauli, DISC word, MBTI, IQ, GTQ, TIKI),
      - top-ranked strength,
      - PAPI scale & score,
      then join to dimension tables for human-readable names (directorate, role, grade).

    Output columns:
      employee_id, full_name, directorate, role, grade, score, pauli,
      disc_1, disc_2, mbti_1, mbti_2, mbti_3, mbti_4,
      iq, gtq, tiki, strengths, scale_code, papi_scores
    """
    assert schema.replace("_", "").isalnum(), "Invalid schema name"

    qry = f"""
    WITH
    employees_base AS (
      SELECT
        e.employee_id,
        e.fullname AS full_name,
        e.nip,
        e.directorate_id,
        e.position_id,
        e.grade_id
      FROM {schema}.employees e
    ),
    competencies_latest AS (
      SELECT DISTINCT ON (cy.employee_id)
             cy.employee_id,
             cy.score,
             cy.year
      FROM {schema}.competencies_yearly cy
      WHERE cy.year IS NOT NULL
      ORDER BY cy.employee_id, cy.year DESC
    ),
    profiles_psych_f AS (
      SELECT
        p.employee_id,
        p.pauli,
        p.disc_word,
        p.mbti,
        p.iq,
        p.gtq,
        p.tiki
      FROM {schema}.profiles_psych p
    ),
    strengths_rank AS (
      SELECT
        s.employee_id,
        s.rank,
        s.theme,
        ROW_NUMBER() OVER (PARTITION BY s.employee_id ORDER BY s.rank ASC, s.theme) AS rn
      FROM {schema}.strengths s
    ),
    strengths_top AS (
      SELECT employee_id, theme
      FROM strengths_rank
      WHERE rn = 1
    ),
    papi_f AS (
      SELECT
        ps.employee_id,
        ps.scale_code,
        ps.score AS papi_scores
      FROM {schema}.papi_scores ps
    ),
    -- Dimension CTEs: map IDs to display names
    dim_directorates_f AS (
      SELECT directorate_id, name AS directorate
      FROM {schema}.dim_directorates
    ),
    dim_positions_f AS (
      SELECT position_id, name AS role
      FROM {schema}.dim_positions
    ),
    dim_grades_f AS (
      SELECT grade_id, name AS grade
      FROM {schema}.dim_grades
    ),
    emp_with_profiles AS (
      SELECT
        eb.employee_id,
        eb.full_name,
        eb.nip,
        eb.directorate_id,
        eb.position_id,
        eb.grade_id,
        pp.pauli,
        pp.disc_word,
        pp.mbti,
        pp.iq,
        pp.gtq,
        pp.tiki
      FROM employees_base eb
      LEFT JOIN profiles_psych_f pp USING (employee_id)
    ),
    emp_with_strengths AS (
      SELECT
        ewp.*,
        st.theme AS strengths
      FROM emp_with_profiles ewp
      LEFT JOIN strengths_top st USING (employee_id)
    ),
    emp_with_papi AS (
      SELECT
        ews.*,
        pf.scale_code,
        pf.papi_scores
      FROM emp_with_strengths ews
      LEFT JOIN papi_f pf USING (employee_id)
    ),
    -- Join human-readable names from dimensions
    emp_with_names AS (
      SELECT
        ewp.*,
        dd.directorate,
        dp.role,
        dg.grade
      FROM emp_with_papi ewp
      LEFT JOIN dim_directorates_f dd ON dd.directorate_id = ewp.directorate_id
      LEFT JOIN dim_positions_f    dp ON dp.position_id    = ewp.position_id
      LEFT JOIN dim_grades_f       dg ON dg.grade_id       = ewp.grade_id
    ),
    -- Parse MBTI/DISC into separate columns
    parsed_splits AS (
      SELECT
        ewp.employee_id,
        NULLIF(split_part(ewp.disc_word, '-', 1), '') AS disc_1,
        NULLIF(split_part(ewp.disc_word, '-', 2), '') AS disc_2,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'I' THEN 'Introversion'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'E' THEN 'Extraversion'
          ELSE NULL
        END AS mbti_1,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'S' THEN 'Sensing'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'N' THEN 'Intuition'
          ELSE NULL
        END AS mbti_2,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'T' THEN 'Thinking'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'F' THEN 'Feeling'
          ELSE NULL
        END AS mbti_3,
        CASE
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'J' THEN 'Judging'
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'P' THEN 'Perceiving'
          ELSE NULL
        END AS mbti_4
      FROM emp_with_names ewp
    ),
    final_raw AS (
      SELECT
        ewp.employee_id,
        ewp.full_name,
        ewp.directorate,  -- nama direktorat
        ewp.role,         -- nama role/position
        ewp.grade,        -- nama grade
        COALESCE(cl.score, 0)  AS score,
        COALESCE(ewp.pauli, 0) AS pauli,
        ps.disc_1,
        ps.disc_2,
        ps.mbti_1,
        ps.mbti_2,
        ps.mbti_3,
        ps.mbti_4,
        COALESCE(ewp.iq,  0)   AS iq,
        COALESCE(ewp.gtq, 0)   AS gtq,
        COALESCE(ewp.tiki,0)   AS tiki,
        ewp.strengths          AS strengths,
        ewp.scale_code,
        COALESCE(ewp.papi_scores, 0) AS papi_scores
      FROM emp_with_names ewp
      LEFT JOIN competencies_latest cl USING (employee_id)
      LEFT JOIN parsed_splits     ps USING (employee_id)
    )
    SELECT DISTINCT
      employee_id,
      full_name,
      directorate,
      role,
      grade,
      score,
      pauli,
      disc_1,
      disc_2,
      mbti_1,
      mbti_2,
      mbti_3,
      mbti_4,
      iq,
      gtq,
      tiki,
      strengths,
      scale_code,
      papi_scores
    FROM final_raw
    ORDER BY employee_id, full_name;
    """

    # ===== Connection to Supabase =====
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD,
        sslmode="require",
        connect_timeout=30,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(qry)  # tidak ada parameter
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()

# ==========================    
# ===== Create Function to Calculate TV-TGV =====
# ==========================
# PAPI letters used (K & Z are handled with special formulas) 
PAPI_LETTERS = ['A','B','C','D','E','F','G','I','K','L','N','O','P','R','S','T','V','W','X','Z']

# Create function to calculate TV TGV
def calculate_tv_tgv(df_employee_benchmarks: pd.DataFrame,
                     df_all_employees: pd.DataFrame,
                     round_digits: int = 2) -> pd.DataFrame:
    """
    Returns a DataFrame containing, per employee:
      employee_id, full_name, directorate, role, grade,
      tv_pauli, tv_iq, tv_gtq, tv_tiki, tv_disc, tv_mbti, tv_strengths,
      tv_papi_kostick,
      tgv_motivation_drive, tgv_leadership_influence, tgv_creativity_innovation,
      tgv_social_orientation, tgv_adaptability_stress, tgv_conscient_reliable,
      tgv_cognitive, tgv_culture_values
    """

    # ---------------------------
    # Helper
    # ---------------------------
    def _mode_nonnull(s: pd.Series):
        vc = s.dropna().value_counts()
        return vc.index[0] if not vc.empty else None

    def _safe_pct(user_val, bench_val):
        if pd.isna(bench_val) or bench_val == 0:
            return np.nan
        return (user_val / bench_val) * 100.0

    def _papi_letter(x):
        if pd.isna(x):
            return None
        s = str(x).upper().strip()
        m = re.search(r'([A-Z])$', s)
        if m and m.group(1) in PAPI_LETTERS:
            return m.group(1)
        m = re.search(r'PAPI[_\-\s]?([A-Z])', s)
        if m and m.group(1) in PAPI_LETTERS:
            return m.group(1)
        return None

    def _disc_has(df: pd.DataFrame, style: str) -> pd.Series:
        return ((df['disc_1'].astype(str) == style) | (df['disc_2'].astype(str) == style)).astype(int) * 100.0

    def _strength_is(df: pd.DataFrame, theme: str) -> pd.Series:
        return (df['strengths'].astype(str) == theme).astype(int) * 100.0

    # ---------------------------
    # Benchmark single-row per employee (avoid duplicates from long PAPI)
    # ---------------------------
    bench_emp = df_employee_benchmarks.drop_duplicates(subset=['employee_id']).copy()

    # Numeric medians  
    med_pauli = pd.to_numeric(bench_emp['pauli'], errors='coerce').median()
    med_iq    = pd.to_numeric(bench_emp['iq'],    errors='coerce').median()
    med_gtq   = pd.to_numeric(bench_emp['gtq'],   errors='coerce').median()
    med_tiki  = pd.to_numeric(bench_emp['tiki'],  errors='coerce').median()

    # Categorical modes (reference for tv_disc / tv_mbti / tv_strengths)
    bench_disc1 = _mode_nonnull(bench_emp['disc_1'])
    bench_disc2 = _mode_nonnull(bench_emp['disc_2'])
    bench_mbti1 = _mode_nonnull(bench_emp['mbti_1'])
    bench_mbti2 = _mode_nonnull(bench_emp['mbti_2'])
    bench_mbti3 = _mode_nonnull(bench_emp['mbti_3'])
    bench_mbti4 = _mode_nonnull(bench_emp['mbti_4'])
    bench_strengths = _mode_nonnull(bench_emp['strengths'])

    # PAPI medians per letter (use long form)
    bench_papi = df_employee_benchmarks[['scale_code','papi_scores']].copy()
    bench_papi['letter'] = bench_papi['scale_code'].map(_papi_letter)
    bench_papi['papi_scores'] = pd.to_numeric(bench_papi['papi_scores'], errors='coerce')
    papi_medians = (bench_papi.dropna(subset=['letter'])
                              .groupby('letter')['papi_scores']
                              .median())

    # ---------------------------
    # Base data: one row per employee from all_employees
    # ---------------------------
    base_cols = ['employee_id','full_name','directorate','role','grade',
                 'pauli','iq','gtq','tiki','disc_1','disc_2',
                 'mbti_1','mbti_2','mbti_3','mbti_4','strengths']
    base = (df_all_employees[base_cols]
            .drop_duplicates(subset=['employee_id'])
            .copy())

    for c in ['pauli','iq','gtq','tiki']:
        base[c] = pd.to_numeric(base[c], errors='coerce')

    # --- TV numeric ---
    base['tv_pauli'] = base['pauli'].apply(lambda x: _safe_pct(x, med_pauli)).clip(upper=100)
    base['tv_iq']    = base['iq'].apply(   lambda x: _safe_pct(x, med_iq)).clip(upper=100)
    base['tv_gtq']   = base['gtq'].apply(  lambda x: _safe_pct(x, med_gtq)).clip(upper=100)
    base['tv_tiki']  = base['tiki'].apply( lambda x: _safe_pct(x, med_tiki)).clip(upper=100)

    # --- TV DISC (0/50/100) ---
    def _tv_disc_row(row):
        match = int(str(row['disc_1']) == str(bench_disc1)) + int(str(row['disc_2']) == str(bench_disc2))
        return (match / 2.0) * 100.0
    base['tv_disc'] = base.apply(_tv_disc_row, axis=1)

    # --- TV MBTI (0/25/50/75/100) ---
    def _tv_mbti_row(row):
        m = 0
        m += int(str(row['mbti_1']) == str(bench_mbti1))
        m += int(str(row['mbti_2']) == str(bench_mbti2))
        m += int(str(row['mbti_3']) == str(bench_mbti3))
        m += int(str(row['mbti_4']) == str(bench_mbti4))
        return (m / 4.0) * 100.0
    base['tv_mbti'] = base.apply(_tv_mbti_row, axis=1)

    # --- TV strengths (0/100) vs mode benchmark ---
    base['tv_strengths'] = (base['strengths'].astype(str) == str(bench_strengths)).astype(int) * 100.0

    # ---------------------------
    # PAPI user (long -> wide), then compute tv_papi_*
    # ---------------------------
    all_papi = df_all_employees[['employee_id','scale_code','papi_scores']].copy()
    all_papi['letter'] = all_papi['scale_code'].map(_papi_letter)
    all_papi['papi_scores'] = pd.to_numeric(all_papi['papi_scores'], errors='coerce')

    user_papi_wide = (all_papi.dropna(subset=['letter'])
                      .pivot_table(index='employee_id', columns='letter',
                                   values='papi_scores', aggfunc='mean'))  # jika duplikat, ambil mean

    tv_papi = pd.DataFrame(index=user_papi_wide.index)
    for L in PAPI_LETTERS:
        col = f'tv_papi_{L}'
        if L in user_papi_wide.columns and L in papi_medians.index:
            bench = papi_medians[L]
            if L in ['K','Z']:
                tv_papi[col] = user_papi_wide[L].apply(
                    lambda x: np.nan if pd.isna(bench) or bench == 0 else ((2*bench - x)/bench)*100.0
                )
            else:
                tv_papi[col] = user_papi_wide[L].apply(lambda x: _safe_pct(x, bench))
        else:
            tv_papi[col] = np.nan
    tv_papi = tv_papi.reset_index()

    # Merge tv_papi back to base 
    out = base.merge(tv_papi, on='employee_id', how='left')

    # Kostick = average of all tv_papi_*   
    papi_cols = [c for c in out.columns if c.startswith('tv_papi_')]
    out['tv_papi_kostick'] = out[papi_cols].mean(axis=1, skipna=True)

    # ---------------------------
    # TGV components (indicator 0/100 + numeric TVs)
    # ---------------------------
    # DISC indicators
    disc_dom  = _disc_has(out, 'Dominant')
    disc_inf  = _disc_has(out, 'Influencer')
    disc_ste  = _disc_has(out, 'Steadiness')
    disc_con  = _disc_has(out, 'Conscientious')

    # MBTI indicators
    # Interpretation: match benchmark on EI axis (mbti_1). To force "Extroversion", 
    # replace this with (out['mbti_1'].eq('Extroversion').astype(int)*100.0)
    mbti1_ei  = (out['mbti_1'].astype(str) == str(bench_mbti1)).astype(int) * 100.0
    mbti2_int = (out['mbti_2'].astype(str).str.lower() == 'intuition').astype(int) * 100.0

    # Strength indicators
    s_Achiever        = _strength_is(out, 'Achiever')
    s_Arranger        = _strength_is(out, 'Arranger')
    s_Command         = _strength_is(out, 'Command')
    s_SelfAssurance   = _strength_is(out, 'Self-Assurance')
    s_Developer       = _strength_is(out, 'Developer')
    s_Futuristic      = _strength_is(out, 'Futuristic')
    s_Ideation        = _strength_is(out, 'Ideation')
    s_Communication   = _strength_is(out, 'Communication')
    s_Woo             = _strength_is(out, 'Woo')
    s_Relator         = _strength_is(out, 'Relator')
    s_Adaptability    = _strength_is(out, 'Adaptability')
    s_Deliberative    = _strength_is(out, 'Deliberative')
    s_Discipline      = _strength_is(out, 'Discipline')
    s_Connectedness   = _strength_is(out, 'Connectedness')
    s_Analytical      = _strength_is(out, 'Analytical')
    s_StrengthsTheme  = _strength_is(out, 'Strengths')   
    s_Belief          = _strength_is(out, 'Belief')

    # Helper to average components (ignore NaN)     
    def _avg(cols):
        return pd.concat(cols, axis=1).mean(axis=1, skipna=True)

    out['tgv_motivation_drive'] = _avg([out['tv_pauli'], out['tv_papi_A'], s_Achiever])
    out['tgv_leadership_influence'] = _avg([
        mbti1_ei, disc_dom, out['tv_papi_L'], out['tv_papi_P'],
        s_Arranger, s_Command, s_SelfAssurance, s_Developer
    ])
    out['tgv_creativity_innovation'] = _avg([mbti2_int, out['tv_papi_Z'], s_Futuristic, s_Ideation])
    out['tgv_social_orientation']    = _avg([disc_inf, out['tv_papi_S'], s_Communication, s_Woo, s_Relator])
    out['tgv_adaptability_stress']   = _avg([disc_ste, out['tv_papi_T'], out['tv_papi_E'], s_Adaptability])
    out['tgv_conscient_reliable']    = _avg([disc_con, out['tv_papi_C'], out['tv_papi_D'], s_Deliberative, s_Discipline])
    out['tgv_cognitive']             = _avg([out['tv_iq'], out['tv_gtq'], out['tv_tiki'],
                                             out['tv_papi_I'], s_Connectedness, s_Analytical, s_StrengthsTheme])
    out['tgv_culture_values']        = s_Belief  # hanya 1 komponen

    # ---------------------------
    # Finishing: rounding & column order
    # ---------------------------
    # final column order
    wanted = [
        'employee_id','full_name','directorate','role','grade',
        'tv_pauli','tv_iq','tv_gtq','tv_tiki','tv_disc','tv_mbti','tv_strengths','tv_papi_kostick',
        'tgv_motivation_drive','tgv_leadership_influence','tgv_creativity_innovation',
        'tgv_social_orientation','tgv_adaptability_stress','tgv_conscient_reliable',
        'tgv_cognitive','tgv_culture_values'
    ]

    # round numeric outputs
    num_cols = [c for c in wanted if c not in ['employee_id','full_name','directorate','role','grade']]
    out[wanted] = out[wanted]  # pastikan kolom ada
    out[num_cols] = out[num_cols].round(round_digits)

    return out[wanted]

# ==========================
# ===== Formatting Output TV-TGV =====
# ==========================
def formatting_tv_tgv(tv_tgv: pd.DataFrame, round_digits: int = 2,
                      w_tv: float = 0.60, w_tgv: float = 0.40) -> pd.DataFrame:
    """
    Build a per-employee summary of TV & TGV.
    final_match_rate = w_tv * tv_match_rate + w_tgv * tgv_match_rate (defaults 0.60 / 0.40).
    If one component is NaN, final_match_rate equals the available component.
    """
    tv_map = {
        'tv_pauli': 'Pauli',
        'tv_mbti': 'MBTI',
        'tv_disc': 'DISC',
        'tv_iq': 'IQ',
        'tv_gtq': 'GTQ',
        'tv_tiki': 'TIKI',
        'tv_strengths': 'CliftonStrengths',
        'tv_papi_kostick': 'PAPI Kostick',
    }
    tgv_map = {
        'tgv_motivation_drive': 'Motivation & Drive',
        'tgv_leadership_influence': 'Leadership & Influence',
        'tgv_creativity_innovation': 'Creativity & Innovation Orientation',
        'tgv_social_orientation': 'Social Orientation & Collaboration',
        'tgv_adaptability_stress': 'Adaptability & Stress Tolerance',
        'tgv_conscient_reliable': 'Conscientiousness & Reliability',
        'tgv_cognitive': 'Cognitive Complexity & Problem-Solving',
        'tgv_culture_values': 'Cultural & Values Urgency',
    }

    tv_cols  = list(tv_map.keys())
    tgv_cols = list(tgv_map.keys())

    # Ensure all required columns exist; if missing, create as NaN so means won’t error
    for c in tv_cols + tgv_cols:
        if c not in tv_tgv.columns:
            tv_tgv[c] = np.nan

    def non_zero_tgv_names(row):
        names = []
        for c in tgv_cols:
            val = row.get(c)
            if pd.notna(val) and float(val) != 0.0:
                names.append(tgv_map[c])
        return names

    tv_match  = tv_tgv[tv_cols].mean(axis=1, skipna=True)
    tgv_match = tv_tgv[tgv_cols].mean(axis=1, skipna=True)

    out = pd.DataFrame({
        'employee_id'   : tv_tgv['employee_id'],
        'full_name'     : tv_tgv['full_name'],
        'directorate'   : tv_tgv['directorate'],
        'role'          : tv_tgv['role'],
        'grade'         : tv_tgv['grade'],
        'tgv_name'      : tv_tgv.apply(non_zero_tgv_names, axis=1),
        'tv_name'       : [list(tv_map.values())] * len(tv_tgv),
        'tv_match_rate' : tv_match.round(round_digits),
        'tgv_match_rate': tgv_match.round(round_digits),
    })

    # Weighted final score (60% TV, 40% TGV by default); if either side is NaN, use the other
    def weighted_final(tv_val, tgv_val):
        if pd.isna(tv_val) and pd.isna(tgv_val):
            return np.nan
        if pd.isna(tv_val):
            return tgv_val
        if pd.isna(tgv_val):
            return tv_val
        return w_tv * tv_val + w_tgv * tgv_val

    out['final_match_rate'] = [
        weighted_final(tv, tg) for tv, tg in zip(out['tv_match_rate'], out['tgv_match_rate'])
    ]
    out['final_match_rate'] = out['final_match_rate'].round(round_digits)

    return out

# ==========================
# ===== Call Client API Groq =====
# ==========================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ==========================
# ===== Define App =====
# ==========================
st.set_page_config(page_title="AI Talent App", layout="wide")
st.title("AI Talent App & Dashboard")

st.markdown("---")

# ==========================
# ===== Fetch List Employee Benchmark to Dropdown =====
# ==========================
def get_conn():
    return psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD,
        sslmode="require",
    )

@st.cache_data(ttl=300)
def fetch_employee_ids_rating5():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT employee_id
                    FROM performance_yearly
                    WHERE rating = 5
                    ORDER BY employee_id
                """)
                rows = cur.fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        st.warning(f"Fail to fetch employee_id: {e}")
        return []

# ==========================
# ============ PDF Builder ============
# ==========================
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
def build_pdf(role_name, job_level, role, jr, jd, kc):
    """PDF ATS-style: bold heading, good bullet, justify text"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm, topMargin=18*mm, bottomMargin=18*mm
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleATS", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=16, leading=19, spaceAfter=8, alignment=TA_LEFT
    )
    h2 = ParagraphStyle(
        "H2ATS", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=10, spaceAfter=6, alignment=TA_LEFT
    )
    meta = ParagraphStyle(
        "MetaATS", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10.5, leading=13, spaceAfter=2, alignment=TA_LEFT
    )
    body = ParagraphStyle(
        "BodyATS", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10.5, leading=14,
        alignment=TA_JUSTIFY,
        leftIndent=12,
        bulletIndent=0,
        spaceBefore=1, spaceAfter=1
    )

    story = []
    story.append(Paragraph("AI Job Description", title))
    story.append(Paragraph(f"<b>Role Name:</b> {role_name}", meta))
    story.append(Paragraph(f"<b>Job Level:</b> {job_level}", meta))
    story.append(Paragraph(f"<b>Role:</b> {role}", meta))
    story.append(Spacer(1, 6))

    # Job Requirements
    story.append(Paragraph("Job Requirements", h2))
    jr_list = jr if isinstance(jr, list) else [str(jr)]
    for it in jr_list:
        story.append(Paragraph(str(it), body, bulletText="•"))
    story.append(Spacer(1, 6))

    # Job Descriptions
    story.append(Paragraph("Job Descriptions", h2))
    jd_text = jd if isinstance(jd, str) else str(jd)
    story.append(Paragraph(jd_text, body))
    story.append(Spacer(1, 6))

    # Key Competencies
    story.append(Paragraph("Key Competencies", h2))
    kc_list = kc if isinstance(kc, list) else [str(kc)]
    for it in kc_list:
        story.append(Paragraph(str(it), body, bulletText="•"))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# ==========================
# ============ UI: AI Talent App Dashboard ============
# ==========================
st.subheader("AI Talent App Dashboard")

SENTINEL = "— Pilih —"
role_options = ["Brand Executive", "Data Analyst", "Finance Office", "HRBP", "Sales Supervisor", "Supply Planner"]
role_options_ui  = [SENTINEL] + role_options
level_options_ui = [SENTINEL, "III", "IV", "V"]

col1, col2, col3 = st.columns(3)
with col1:
    role_name = st.selectbox("Role Name *", role_options_ui, index=0)
with col2:
    job_level = st.selectbox("Job Level *", level_options_ui, index=0)
with col3:
    role = st.selectbox("Role *", role_options_ui, index=0)

role_purposes = st.text_area(
    "Role purposes *",
    placeholder="1–2 sentences to describe role outcome",
    height=100,
)

with st.spinner("Load list benchmarks employee_id..."):
    emp_options = fetch_employee_ids_rating5()
selected_emp = st.multiselect(
    "Employee Benchmarking *",
    options=emp_options,
    placeholder="Choose employee benchmarking (min 1, max 3)",
    max_selections=3
)
st.caption(f"Chosen: {len(selected_emp)} employee")

# ==========================
# ================== BACKEND CALL : AI GENERATION & TV-TGV MATCH RATE ==================
# ==========================
if st.button("Generate Job Description & TV-TGV Match Rate", type="primary", key="jd_btn"):
    # ---- VALIDATION: all required ----
    errors = []
    if role_name == SENTINEL: errors.append("Role Name must be selected.")
    if job_level == SENTINEL: errors.append("Job Level must be selected.")
    if role == SENTINEL: errors.append("Role must be selected.")
    if not role_purposes.strip(): errors.append("Role purposes must be selected.")
    # You must select at least one employee if the list is available
    if emp_options and len(selected_emp) == 0:
        errors.append("Choose minimum 1 Employee Benchmarking.")

    if errors:
        st.error("Check the inputs:")
        for e in errors:
            st.write(f"- {e}")
    else:
        # ====== Call AI only when all inputs valid ======
        with st.spinner("Ask AI to Create Job Description..."):
            try:
                prompt = f"""
You are an expert HR recruiter. Create a concise AI Job Description for:
- Role Name: {role_name}
- Job Level: {job_level}
- Role: {role}
Role purpose: {role_purposes}

Return ONLY a valid JSON object (no markdown, no code fences) with this exact schema and keys:
{{
  "Job Requirements": ["<bullet 1>", "<bullet 2>", "..."],
  "Job Descriptions": "<1-3 sentences>",
  "Key Competencies": ["<tool/skill 1>", "<tool/skill 2>", "..."]
}}

Constraints:
- "Job Requirements": 5–10 concise bullets, in English.
- "Job Descriptions": 1–3 sentences, crisp and outcome-oriented.
- "Key Competencies": list relevant tools/platforms tailored to the role.
"""
                resp = client.responses.create(
                    model=MODEL,
                    input=prompt,
                    temperature=0.2,
                )
                raw = resp.output_text.strip()

                try:
                    data = json.loads(raw)
                except Exception:
                    start = raw.find("{"); end = raw.rfind("}") + 1
                    data = json.loads(raw[start:end])

                jr = data.get("Job Requirements", [])
                jd = data.get("Job Descriptions", "")
                kc = data.get("Key Competencies", [])

                # ---- Show (no table)
                st.markdown("### Job Requirements")
                if isinstance(jr, list) and jr:
                    for item in jr: st.write(f"• {item}")
                else:
                    st.write("-")

                st.markdown("### Job Descriptions")
                st.write(jd if isinstance(jd, str) and jd else "-")

                st.markdown("### Key Competencies")
                if isinstance(kc, list) and kc:
                    for item in kc: st.write(f"• {item}")
                else:
                    st.write("-")

                # ---- Download PDF (ATS style)
                pdf_bytes = build_pdf(role_name, job_level, role, jr, jd, kc)
                st.download_button(
                    "Download PDF (ATS style)",
                    data=pdf_bytes,
                    file_name=f"AI_JD_{role_name}_{job_level}.pdf",
                    mime="application/pdf",
                )

                st.markdown("---")

                # Show Employee
                st.subheader('Talent Rank List')
                st.caption("Chosen Employee Benchmarking:")
                st.write(selected_emp)

                # Load Dataset Employee
                df_employee_benchmarks = employees_benchmark_data(selected_emp, schema="public")
                df_all_employees = get_all_employee_data(schema="public")

                st.write('### Table Employee Benchmarks')
                st.dataframe(df_employee_benchmarks)

                # Calculate TV-TGV
                tv_tgv = calculate_tv_tgv(df_employee_benchmarks, df_all_employees)
                formatted_tv_tgv = formatting_tv_tgv(tv_tgv, round_digits=2)
                final_results = formatted_tv_tgv.sort_values(by='final_match_rate', ascending=False)
                final_results = final_results.rename(columns={'employee_id':'Employee ID',
                                                              'full_name':'Full Name',
                                                              'directorate':'Directorate',
                                                              'role':'Role',
                                                              'grade':'Grade',
                                                              'tgv_name':'TGV Name',
                                                              'tv_name':'TV Name',
                                                              'tv_match_rate':'TV Match Rate',
                                                              'tgv_match_rate':'TGV Match Rate',
                                                              'final_match_rate':'Final Match Rate'})
                st.write('### Talent Rank List')
                st.dataframe(final_results)

                st.write('### Data Visualization')
                # --- Summary Match vs Not Match (threshold 60%) ---
                thr = 60.0
                series = formatted_tv_tgv["final_match_rate"].dropna()   # Compute for records with non-empty values only
                total = int(series.shape[0])

                matched_cnt = int((series >= thr).sum())
                not_matched_cnt = int((series <  thr).sum())

                # Avoid div-by-zero
                if total == 0:
                    matched_pct = not_matched_pct = 0.0
                else:
                    matched_pct     = matched_cnt / total * 100.0
                    not_matched_pct = not_matched_cnt / total * 100.0

                st.write('#### Data Statistics')
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Matched (≥60%)")
                    st.write(f"{matched_cnt} ({matched_pct:.1f}%)")

                with c2:
                    st.subheader("Not Matched (<60%)")
                    st.write(f"{not_matched_cnt} ({not_matched_pct:.1f}%)")
                
                # --- Top Strengths All Employee ---
                st.write('#### Top Strengths All Employee')
                s = df_all_employees['strengths'].astype(str).str.strip()
                s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})  # empty -> NaN
                counts = s.dropna().value_counts()  # sort desc by default

                cats = counts.index.tolist()   # category (strengths)
                vals = counts.values.tolist()  # counts

                # Plot horizontal bar
                fig_bar_strengths = go.Figure(go.Bar(
                    x=vals,
                    y=cats,
                    orientation='h',
                    text=vals,                 # data labels
                    textposition='outside',
                    hovertemplate='%{y}: %{x}<extra></extra>'
                ))

                # Sort so that the largest category appears first
                fig_bar_strengths.update_yaxes(autorange='reversed')

                fig_bar_strengths.update_layout(
                    xaxis_title='Count',
                    yaxis_title='Strength',
                    margin=dict(l=80, r=80, t=60, b=40)
                )

                st.plotly_chart(fig_bar_strengths, use_container_width=True)
                
                # --- Radar Chart : All Employee vs Benchmark Employee ---
                metrics = ["pauli", "iq", "gtq"]

                # Make sure the columns are numeric
                df_all_num = df_all_employees.copy()
                df_bench_num = df_employee_benchmarks.copy()
                for c in metrics:
                    df_all_num[c] = pd.to_numeric(df_all_num[c], errors="coerce")
                    df_bench_num[c] = pd.to_numeric(df_bench_num[c], errors="coerce")

                # Aggregation
                median_all = df_all_num[metrics].median(skipna=True)
                median_bench = df_bench_num[metrics].median(skipna=True)

                # Auto radial scale (add a little space above the maximum value)
                max_val = float(np.nanmax([median_all.max(), median_bench.max()]))
                radial_max = max_val * 1.1 if np.isfinite(max_val) else 1.0

                # Plot Radar Chart
                fig_radar_chart = go.Figure()

                fig_radar_chart.add_trace(go.Scatterpolar(
                    r=median_all.values,
                    theta=metrics,
                    fill='toself',
                    name='All Employees (Median)'
                ))

                fig_radar_chart.add_trace(go.Scatterpolar(
                    r=median_bench.values,
                    theta=metrics,
                    fill='toself',
                    name='Benchmark (Median)'
                ))

                fig_radar_chart.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, radial_max])
                    ),
                    showlegend=True
                )

                st.write('#### Radar Chart: All vs Benchmark (Pauli, IQ, GTQ, TIKI)')
                st.plotly_chart(fig_radar_chart, use_container_width=True)

                # --- AI Insight (Groq) from Top 5 ---
                try:
                    # If final_results is missing or empty, fail gracefully
                    if "final_match_rate" in locals() or "final_results" in locals():
                        pass  # only for linter

                    if final_results.empty:
                        st.info("No final_results to analyze for AI summary.")
                    else:
                        top5 = final_results.head(5).copy()

                        # Prepare a concise context for the AI (avoid overly long payloads)
                        cols_for_ai = ["Employee ID", "Full Name", "Role", "Grade",
                                    "Final Match Rate", "TV Match Rate", "TGV Match Rate", "TGV Name"]
                        # “Ensure required columns exist
                        cols_for_ai = [c for c in cols_for_ai if c in top5.columns]

                        top5_for_ai = top5[cols_for_ai].copy()

                        # Pastikan tipe list/objek jadi string yang ringkasConvert list/object types into short string representations
                        if "TGV Name" in top5_for_ai.columns:
                            top5_for_ai["TGV Name"] = top5_for_ai["TGV Name"].apply(
                                lambda x: ", ".join(x) if isinstance(x, (list, tuple, set)) else (str(x) if pd.notna(x) else "")
                            )

                        # (Optional) Round numbers for cleaner output
                        for c in ["Final Match Rate", "TV Match Rate", "TGV Match Rate"]:
                            if c in top5_for_ai.columns:
                                top5_for_ai[c] = pd.to_numeric(top5_for_ai[c], errors="coerce").round(2)

                        context_json = json.dumps(top5_for_ai.to_dict(orient="records"))[:5000]  # batasi panjang

                        ai_prompt = f"""
                You are an expert HR analyst. Using the following top employees data (JSON):

                {context_json}

                Write a single cohesive paragraph in English with 4–5 sentences that provides:
                - Summary insights explaining why certain employees rank highest / have a high match rate,
                - Common drivers behind high alignment (e.g., cognitive/aptitude alignment, consistent DISC/MBTI signals, strengths fit),
                - Any notable distinguishing factors among the top profiles,
                - Practical implications for role fit or deployment.

                Avoid listing raw numbers; synthesize the reasons. Keep it under 120 words.
                """

                        # Call Groq
                        groq_resp = client.responses.create(
                            model=MODEL,
                            input=ai_prompt,
                            temperature=0.2,
                        )
                        ai_text = groq_resp.output_text.strip()

                        # Show AI Results
                        st.write("#### AI Summary Insight")
                        st.write(ai_text)

                except Exception as e:
                    st.warning(f"AI summary failed: {e}")
                
            except Exception as e:
                st.error(f"Error found: {e}")