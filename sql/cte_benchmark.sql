WITH                                                          -- Begin CTE section
    ids AS (                                                       -- CTE: expand provided IDs
      SELECT UNNEST(%s::text[]) AS employee_id                    -- Turn the text[] parameter into one row per employee_id
    ),                                                            -- Close CTE; continue to next
    employees_base AS (                                           -- CTE: base employees limited to requested IDs
      SELECT                                                      -- Select columns needed downstream
        e.employee_id,                                            -- Employee primary key
        e.fullname AS full_name,                                  -- Human-readable full name
        e.nip,                                                    -- Internal employee number
        e.directorate_id,                                         -- FK to directorates dimension
        e.position_id,                                            -- FK to positions (role) dimension
        e.grade_id                                                -- FK to grades dimension
      FROM public.employees e                                   -- Source table: employees master
      JOIN ids i ON i.employee_id = e.employee_id::text           -- Keep only employees present in the input ID list
    ),                                                            -- Close CTE; continue
    competencies_latest AS (                                      -- CTE: latest competency score per employee (by year)
      SELECT DISTINCT ON (cy.employee_id)                         -- Keep only the newest row per employee_id
            cy.employee_id,                                      -- Employee key
            cy.score,                                            -- Competency score
            cy.year                                              -- Year of the score
      FROM public.competencies_yearly cy                        -- Source table: yearly competencies
      JOIN ids i ON i.employee_id = cy.employee_id::text          -- Restrict to requested employees
      WHERE cy.year IS NOT NULL                                   -- Ignore rows that don't have a valid year
      ORDER BY cy.employee_id, cy.year DESC                       -- Sort so the newest year comes first per employee
    ),                                                            -- Close CTE; continue
    profiles_psych_f AS (                                         -- CTE: psychological profiles (if any)
      SELECT                                                      -- Select psychometric fields
        p.employee_id,                                            -- Employee key
        p.pauli,                                                  -- Pauli score
        p.disc_word,                                              -- DISC word (e.g., "D-I")
        p.mbti,                                                   -- MBTI type (e.g., "INTJ")
        p.iq,                                                     -- IQ score
        p.gtq,                                                    -- GTQ score
        p.tiki                                                    -- TIKI score
      FROM public.profiles_psych p                              -- Source table: profiles_psych
      JOIN ids i ON i.employee_id = p.employee_id::text           -- Restrict to requested employees
    ),                                                            -- Close CTE; continue
    strengths_rank AS (                                           -- CTE: rank strengths to find the top theme
      SELECT                                                      -- Compute row_number over strengths per employee
        s.employee_id,                                            -- Employee key
        s.rank,                                                   -- Strength rank value
        s.theme,                                                  -- Strength theme label
        ROW_NUMBER() OVER (PARTITION BY s.employee_id             -- Window: restart per employee
                          ORDER BY s.rank ASC, s.theme) AS rn     -- rn=1 marks the top-ranked (ties broken by theme)
      FROM public.strengths s                                   -- Source table: strengths
      JOIN ids i ON i.employee_id = s.employee_id::text           -- Restrict to requested employees
    ),                                                            -- Close CTE; continue
    strengths_top AS (                                            -- CTE: keep only the top strength per employee
      SELECT employee_id, theme                                   -- Output: employee and their top theme
      FROM strengths_rank                                         -- Source: ranked strengths
      WHERE rn = 1                                                -- Filter to top-ranked row
    ),                                                            -- Close CTE; continue
    papi_f AS (                                                   -- CTE: PAPI scores per scale (may be multiple rows)
      SELECT                                                      -- Select PAPI details
        ps.employee_id,                                           -- Employee key
        ps.scale_code,                                            -- PAPI scale identifier
        ps.score AS papi_scores                                   -- PAPI score value
      FROM public.papi_scores ps                                -- Source table: papi_scores
      JOIN ids i ON i.employee_id = ps.employee_id::text          -- Restrict to requested employees
    ),                                                            -- Close CTE; continue
    -- Dimension CTEs: map IDs to display names
    dim_directorates_f AS (                                       -- CTE: directorate names
      SELECT directorate_id, name AS directorate                  -- Map directorate_id -> directorate name
      FROM public.dim_directorates                              -- Source dimension: directorates
    ),                                                            -- Close CTE; continue
    dim_positions_f AS (                                          -- CTE: position/role names
      SELECT position_id, name AS role                            -- Map position_id -> role name
      FROM public.dim_positions                                 -- Source dimension: positions
    ),                                                            -- Close CTE; continue
    dim_grades_f AS (                                             -- CTE: grade names
      SELECT grade_id, name AS grade                              -- Map grade_id -> grade name
      FROM public.dim_grades                                    -- Source dimension: grades
    ),                                                            -- Close CTE; continue
    emp_with_profiles AS (                                        -- CTE: employees enriched with psych profiles
      SELECT                                                      -- Select base employee fields + profile columns
        eb.employee_id,                                           -- Employee key
        eb.full_name,                                             -- Name
        eb.nip,                                                   -- Internal number
        eb.directorate_id,                                        -- FK directorate
        eb.position_id,                                           -- FK position
        eb.grade_id,                                              -- FK grade
        pp.pauli,                                                 -- Pauli (nullable)
        pp.disc_word,                                             -- DISC (nullable)
        pp.mbti,                                                  -- MBTI (nullable)
        pp.iq,                                                    -- IQ (nullable)
        pp.gtq,                                                   -- GTQ (nullable)
        pp.tiki                                                   -- TIKI (nullable)
      FROM employees_base eb                                      -- Source: filtered employees
      LEFT JOIN profiles_psych_f pp USING (employee_id)           -- Optional join: keep employees without profiles
    ),                                                            -- Close CTE; continue
    emp_with_strengths AS (                                       -- CTE: add top strength theme
      SELECT                                                      -- Carry all previous fields and add theme
        ewp.*,                                                    -- All columns from emp_with_profiles
        st.theme AS strengths                                     -- Top strength theme (nullable)
      FROM emp_with_profiles ewp                                  -- Source: previous CTE
      LEFT JOIN strengths_top st USING (employee_id)              -- Optional join: attach top theme if any
    ),                                                            -- Close CTE; continue
    emp_with_papi AS (                                            -- CTE: add PAPI scale/score (may add row multiplicity)
      SELECT                                                      -- Carry all previous fields and add PAPI
        ews.*,                                                    -- All columns from emp_with_strengths
        pf.scale_code,                                            -- PAPI scale code (nullable)
        pf.papi_scores                                            -- PAPI score (nullable)
      FROM emp_with_strengths ews                                 -- Source: previous CTE
      LEFT JOIN papi_f pf USING (employee_id)                     -- Optional join: attach PAPI per scale
    ),                                                            -- Close CTE; continue
    -- Join human-readable names from dimensions
    emp_with_names AS (                                           -- CTE: attach directorate/role/grade names
      SELECT                                                      -- Carry all previous fields + names
        ewp.*,                                                    -- All columns so far
        dd.directorate,                                           -- Directorate display name
        dp.role,                                                  -- Role/position display name
        dg.grade                                                  -- Grade display name
      FROM emp_with_papi ewp                                      -- Source: previous CTE
      LEFT JOIN dim_directorates_f dd ON dd.directorate_id = ewp.directorate_id  -- Join directorate name
      LEFT JOIN dim_positions_f    dp ON dp.position_id    = ewp.position_id     -- Join role name
      LEFT JOIN dim_grades_f       dg ON dg.grade_id       = ewp.grade_id         -- Join grade name
    ),                                                            -- Close CTE; continue
    -- Parse MBTI/DISC into separate columns
    parsed_splits AS (                                            -- CTE: per-employee DISC/MBTI splits
      SELECT                                                      -- Compute derived columns
        ewp.employee_id,                                          -- Employee key
        NULLIF(split_part(ewp.disc_word, '-', 1), '') AS disc_1,  -- DISC first component; empty string -> NULL
        NULLIF(split_part(ewp.disc_word, '-', 2), '') AS disc_2,  -- DISC second component; empty string -> NULL
        CASE                                                      -- MBTI letter 1 -> Introversion/Extraversion
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL  -- Guard for invalid MBTI
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'I' THEN 'Introversion' -- I => Introversion
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),1,1)) = 'E' THEN 'Extraversion' -- E => Extraversion
          ELSE NULL                                               -- Anything else -> NULL
        END AS mbti_1,                                            -- Aliased as mbti_1
        CASE                                                      -- MBTI letter 2 -> Sensing/Intuition
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL  -- Guard for invalid MBTI
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'S' THEN 'Sensing'  -- S => Sensing
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),2,1)) = 'N' THEN 'Intuition'-- N => Intuition
          ELSE NULL                                               -- Anything else -> NULL
        END AS mbti_2,                                            -- Aliased as mbti_2
        CASE                                                      -- MBTI letter 3 -> Thinking/Feeling
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL  -- Guard for invalid MBTI
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'T' THEN 'Thinking' -- T => Thinking
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),3,1)) = 'F' THEN 'Feeling'  -- F => Feeling
          ELSE NULL                                               -- Anything else -> NULL
        END AS mbti_3,                                            -- Aliased as mbti_3
        CASE                                                      -- MBTI letter 4 -> Judging/Perceiving
          WHEN ewp.mbti IS NULL OR length(trim(ewp.mbti)) < 4 THEN NULL  -- Guard for invalid MBTI
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'J' THEN 'Judging'  -- J => Judging
          WHEN UPPER(SUBSTRING(TRIM(ewp.mbti),4,1)) = 'P' THEN 'Perceiving' -- P => Perceiving
          ELSE NULL                                               -- Anything else -> NULL
        END AS mbti_4                                             -- Aliased as mbti_4
      FROM emp_with_names ewp                                     -- Source rows to parse
    ),                                                            -- Close CTE; continue
    -- Assemble raw output (may have duplicates)
    final_raw AS (                                                -- CTE: compose raw result rows
      SELECT                                                      -- Select normalized outputs
        ewp.employee_id,                                          -- Employee key
        ewp.full_name,                                            -- Name
        ewp.directorate,                                          -- Directorate display name
        ewp.role,                                                 -- Role/position display name
        ewp.grade,                                                -- Grade display name
        COALESCE(cl.score, 0)  AS score,                          -- Latest competency score; default 0
        COALESCE(ewp.pauli, 0) AS pauli,                          -- Pauli score; default 0
        ps.disc_1,                                                -- DISC part 1
        ps.disc_2,                                                -- DISC part 2
        ps.mbti_1,                                                -- MBTI dim 1
        ps.mbti_2,                                                -- MBTI dim 2
        ps.mbti_3,                                                -- MBTI dim 3
        ps.mbti_4,                                                -- MBTI dim 4
        COALESCE(ewp.iq,  0)   AS iq,                             -- IQ; default 0
        COALESCE(ewp.gtq, 0)   AS gtq,                            -- GTQ; default 0
        COALESCE(ewp.tiki,0)   AS tiki,                           -- TIKI; default 0
        ewp.strengths          AS strengths,                      -- Top strength theme
        ewp.scale_code,                                           -- PAPI scale code
        COALESCE(ewp.papi_scores, 0) AS papi_scores               -- PAPI score; default 0
      FROM emp_with_names ewp                                     -- Source: names and PAPI attached
      LEFT JOIN competencies_latest cl USING (employee_id)        -- Add the latest competency score
      LEFT JOIN parsed_splits     ps USING (employee_id)          -- Add DISC/MBTI split columns
    )                                                             -- Close CTE; continue
    -- Final select with de-duplication
    SELECT DISTINCT                                               -- Remove duplicate rows in the projection
      employee_id,                                                -- Employee key
      full_name,                                                  -- Name
      directorate,                                                -- Directorate
      role,                                                       -- Role/position
      grade,                                                      -- Grade
      score,                                                      -- Latest competency score
      pauli,                                                      -- Pauli score
      disc_1,                                                     -- DISC part 1
      disc_2,                                                     -- DISC part 2
      mbti_1,                                                     -- MBTI dim 1
      mbti_2,                                                     -- MBTI dim 2
      mbti_3,                                                     -- MBTI dim 3
      mbti_4,                                                     -- MBTI dim 4
      iq,                                                         -- IQ
      gtq,                                                        -- GTQ
      tiki,                                                       -- TIKI
      strengths,                                                  -- Top strength theme
      scale_code,                                                 -- PAPI scale code
      papi_scores                                                 -- PAPI score
    FROM final_raw                                                -- Read from the assembled raw CTE
    ORDER BY employee_id, full_name;                              -- Stable, readable ordering by employee id & name