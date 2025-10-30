WITH                                                         -- Start CTE block
    employees_base AS (                                          -- CTE: base employees (all rows)
      SELECT                                                     -- Select base fields
        e.employee_id,                                           -- Employee primary key
        e.fullname AS full_name,                                 -- Human-readable full name
        e.nip,                                                   -- Internal employee number
        e.directorate_id,                                        -- FK to directorates
        e.position_id,                                           -- FK to positions (role)
        e.grade_id                                               -- FK to grades
      FROM public.employees e                                  -- Source: employees master (entire table)
    ),                                                           -- End CTE employees_base
    competencies_latest AS (                                     -- CTE: pick latest competency per employee
      SELECT DISTINCT ON (cy.employee_id)                        -- Keep one row per employee_id (latest by year)
            cy.employee_id,                                     -- Employee key
            cy.score,                                           -- Competency score
            cy.year                                             -- Year of the score
      FROM public.competencies_yearly cy                       -- Source: competencies by year
      WHERE cy.year IS NOT NULL                                  -- Exclude rows with NULL year
      ORDER BY cy.employee_id, cy.year DESC                      -- Sort so newest year is chosen by DISTINCT ON
    ),                                                           -- End CTE competencies_latest
    profiles_psych_f AS (                                        -- CTE: psychological profiles
      SELECT                                                     -- Select psych fields
        p.employee_id,                                           -- Employee key
        p.pauli,                                                 -- Pauli score
        p.disc_word,                                             -- DISC word (e.g., "D-I")
        p.mbti,                                                  -- MBTI type (e.g., "INTJ")
        p.iq,                                                    -- IQ score
        p.gtq,                                                   -- GTQ score
        p.tiki                                                   -- TIKI score
      FROM public.profiles_psych p                             -- Source: psych profiles table
    ),                                                           -- End CTE profiles_psych_f
    strengths_rank AS (                                          -- CTE: rank strengths per employee
      SELECT                                                     -- Compute row numbers by employee
        s.employee_id,                                           -- Employee key
        s.rank,                                                  -- Rank of the strength
        s.theme,                                                 -- Strength theme
        ROW_NUMBER() OVER (PARTITION BY s.employee_id            -- Row number restart per employee
                          ORDER BY s.rank ASC, s.theme) AS rn    -- rn=1 marks top theme (tie-break by theme)
      FROM public.strengths s                                  -- Source: strengths table
    ),                                                           -- End CTE strengths_rank
    strengths_top AS (                                           -- CTE: keep only top strength per employee
      SELECT employee_id, theme                                  -- Output key + top theme
      FROM strengths_rank                                        -- Source: ranked strengths
      WHERE rn = 1                                               -- Keep top-ranked only
    ),                                                           -- End CTE strengths_top
    papi_f AS (                                                  -- CTE: PAPI scores (may be multiple per employee)
      SELECT                                                     -- Select PAPI fields
        ps.employee_id,                                          -- Employee key
        ps.scale_code,                                           -- PAPI scale code
        ps.score AS papi_scores                                  -- PAPI score value
      FROM public.papi_scores ps                               -- Source: PAPI scores table
    ),                                                           -- End CTE papi_f
    -- Dimension CTEs: map IDs to display names                  -- Note: next CTEs map IDs -> names
    dim_directorates_f AS (                                      -- CTE: directorate names
      SELECT directorate_id, name AS directorate                 -- Map directorate_id to name
      FROM public.dim_directorates                             -- Source: directorates dimension
    ),                                                           -- End CTE dim_directorates_f
    dim_positions_f AS (                                         -- CTE: role/position names
      SELECT position_id, name AS role                           -- Map position_id to role name
      FROM public.dim_positions                                -- Source: positions dimension
    ),                                                           -- End CTE dim_positions_f
    dim_grades_f AS (                                            -- CTE: grade names
      SELECT grade_id, name AS grade                             -- Map grade_id to grade name
      FROM public.dim_grades                                   -- Source: grades dimension
    ),                                                           -- End CTE dim_grades_f
    emp_with_profiles AS (                                       -- CTE: join base with psych profiles
      SELECT                                                     -- Select base + psych fields
        eb.employee_id,                                          -- Employee key
        eb.full_name,                                            -- Full name
        eb.nip,                                                  -- Employee number
        eb.directorate_id,                                       -- FK directorate
        eb.position_id,                                          -- FK position
        eb.grade_id,                                             -- FK grade
        pp.pauli,                                                -- Pauli (nullable)
        pp.disc_word,                                            -- DISC word (nullable)
        pp.mbti,                                                 -- MBTI (nullable)
        pp.iq,                                                   -- IQ (nullable)
        pp.gtq,                                                  -- GTQ (nullable)
        pp.tiki                                                  -- TIKI (nullable)
      FROM employees_base eb                                     -- Source: all employees
      LEFT JOIN profiles_psych_f pp USING (employee_id)          -- Left join: keep employees without profiles
    ),                                                           -- End CTE emp_with_profiles
    emp_with_strengths AS (                                      -- CTE: add top strength theme
      SELECT                                                     -- Carry all cols + theme
        ewp.*,                                                   -- All columns from previous CTE
        st.theme AS strengths                                    -- Top strength theme (nullable)
      FROM emp_with_profiles ewp                                 -- Source: employees with profiles
      LEFT JOIN strengths_top st USING (employee_id)             -- Left join: attach top theme
    ),                                                           -- End CTE emp_with_strengths
    emp_with_papi AS (                                           -- CTE: add PAPI scale and score
      SELECT                                                     -- Carry all cols + PAPI fields
        ews.*,                                                   -- All columns so far
        pf.scale_code,                                           -- PAPI scale code (nullable)
        pf.papi_scores                                           -- PAPI score (nullable)
      FROM emp_with_strengths ews                                -- Source: employees with strengths
      LEFT JOIN papi_f pf USING (employee_id)                    -- Left join: attach PAPI rows
    ),                                                           -- End CTE emp_with_papi
    -- Join human-readable names from dimensions                 -- Note: join human-readable names
    emp_with_names AS (                                          -- CTE: attach names from dimensions
      SELECT                                                     -- Carry all cols + names
        ewp.*,                                                   -- All columns so far
        dd.directorate,                                          -- Directorate name
        dp.role,                                                 -- Role/position name
        dg.grade                                                 -- Grade name
      FROM emp_with_papi ewp                                     -- Source: employees with PAPI
      LEFT JOIN dim_directorates_f dd ON dd.directorate_id = ewp.directorate_id  -- Join directorate name
      LEFT JOIN dim_positions_f    dp ON dp.position_id    = ewp.position_id     -- Join role name
      LEFT JOIN dim_grades_f       dg ON dg.grade_id       = ewp.grade_id         -- Join grade name
    ),                                                           -- End CTE emp_with_names
    -- Parse MBTI/DISC into separate columns                     -- Note: normalize MBTI & DISC
    parsed_splits AS (                                           -- CTE: derive DISC/MBTI columns
      SELECT                                                     -- Compute parsed fields
        ewp.employee_id,                                         -- Employee key
        NULLIF(split_part(ewp.disc_word, '-', 1), '') AS disc_1, -- DISC first component ('' -> NULL)
        NULLIF(split_part(ewp.disc_word, '-', 2), '') AS disc_2, -- DISC second component ('' -> NULL)
        CASE                                                     -- MBTI letter 1 -> Intro/Extra
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL -- Guard for invalid MBTI
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'I' THEN 'Introversion' -- I => Introversion
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'E' THEN 'Extraversion' -- E => Extraversion
          ELSE NULL                                              -- Otherwise NULL
        END AS mbti_1,                                           -- Alias mbti_1
        CASE                                                     -- MBTI letter 2 -> Sens/Intuit
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL -- Guard
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'S' THEN 'Sensing'  -- S => Sensing
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'N' THEN 'Intuition'-- N => Intuition
          ELSE NULL                                              -- Otherwise NULL
        END AS mbti_2,                                           -- Alias mbti_2
        CASE                                                     -- MBTI letter 3 -> Think/Feel
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL -- Guard
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'T' THEN 'Thinking' -- T => Thinking
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'F' THEN 'Feeling'  -- F => Feeling
          ELSE NULL                                              -- Otherwise NULL
        END AS mbti_3,                                           -- Alias mbti_3
        CASE                                                     -- MBTI letter 4 -> Judge/Perceive
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL -- Guard
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'J' THEN 'Judging'  -- J => Judging
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'P' THEN 'Perceiving' -- P => Perceiving
          ELSE NULL                                              -- Otherwise NULL
        END AS mbti_4                                            -- Alias mbti_4
      FROM emp_with_names ewp                                    -- Source rows for parsing
    ),                                                           -- End CTE parsed_splits
    final_raw AS (                                               -- CTE: assemble raw output
      SELECT                                                     -- Select final fields (pre-dedup)
        ewp.employee_id,                                         -- Employee key
        ewp.full_name,                                           -- Full name
        ewp.directorate,  -- directorate name                    -- Directorate name (already resolved)
        ewp.role,         -- role/position name                  -- Role name (already resolved)
        ewp.grade,        -- grade name                          -- Grade name (already resolved)
        COALESCE(cl.score, 0)  AS score,                         -- Latest competency; default 0
        COALESCE(ewp.pauli, 0) AS pauli,                         -- Pauli; default 0
        ps.disc_1,                                                -- DISC part 1
        ps.disc_2,                                                -- DISC part 2
        ps.mbti_1,                                                -- MBTI dim 1
        ps.mbti_2,                                                -- MBTI dim 2
        ps.mbti_3,                                                -- MBTI dim 3
        ps.mbti_4,                                                -- MBTI dim 4
        COALESCE(ewp.iq,  0)   AS iq,                            -- IQ; default 0
        COALESCE(ewp.gtq, 0)   AS gtq,                           -- GTQ; default 0
        COALESCE(ewp.tiki,0)   AS tiki,                          -- TIKI; default 0
        ewp.strengths          AS strengths,                     -- Top strength theme
        ewp.scale_code,                                          -- PAPI scale code
        COALESCE(ewp.papi_scores, 0) AS papi_scores              -- PAPI score; default 0
      FROM emp_with_names ewp                                    -- Source: names + PAPI attached
      LEFT JOIN competencies_latest cl USING (employee_id)       -- Add latest competency by employee
      LEFT JOIN parsed_splits     ps USING (employee_id)         -- Add parsed DISC/MBTI fields
    )                                                            -- End CTE final_raw
    SELECT DISTINCT                                              -- Final projection with deduplication
      employee_id,                                               -- Employee key
      full_name,                                                 -- Full name
      directorate,                                               -- Directorate
      role,                                                      -- Role
      grade,                                                     -- Grade
      score,                                                     -- Competency score
      pauli,                                                     -- Pauli score
      disc_1,                                                    -- DISC part 1
      disc_2,                                                    -- DISC part 2
      mbti_1,                                                    -- MBTI dim 1
      mbti_2,                                                    -- MBTI dim 2
      mbti_3,                                                    -- MBTI dim 3
      mbti_4,                                                    -- MBTI dim 4
      iq,                                                        -- IQ
      gtq,                                                       -- GTQ
      tiki,                                                      -- TIKI
      strengths,                                                 -- Top strength theme
      scale_code,                                                -- PAPI scale code
      papi_scores                                                -- PAPI score
    FROM final_raw                                               -- Read from assembled CTE
    ORDER BY employee_id, full_name;                             -- Stable, readable ordering by employee id and name