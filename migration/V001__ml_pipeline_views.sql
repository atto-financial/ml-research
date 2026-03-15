-- =============================================================================
-- V001__ml_pipeline_views.sql
--
-- Based on V000_current_dump.md (current DB state):
--   All source tables already exist with UUID PKs:
--     users (user_id uuid), features (feature_id uuid, feature_slug varchar),
--     answers (user_id uuid, feature_id uuid, answer jsonb),
--     loan_summary_statuses (user_id uuid PK, loan_status enum, payoff_score int)
--
--   Missing: staging/intermediate/mart VIEWS needed by the Dagster ML pipeline.
--   This migration creates only those views.
-- =============================================================================


-- ─────────────────────────────────────────────
-- 1. STAGING VIEWS
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW stg_users AS
    SELECT user_id FROM users;

CREATE OR REPLACE VIEW stg_features AS
    SELECT feature_id, feature_slug FROM features;

CREATE OR REPLACE VIEW stg_answers AS
    SELECT user_id, feature_id, answer FROM answers;

CREATE OR REPLACE VIEW stg_loan_summary AS
    SELECT user_id, loan_status, payoff_score FROM loan_summary_statuses;


-- ─────────────────────────────────────────────
-- 2. stg_loan_features_extracted
--    Unpacks JSONB answers into flat fht/kmsi columns.
--    Mirrors dbt/staging/stg_loan_features_extracted.sql
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW stg_loan_features_extracted AS
SELECT
    u.user_id,
    CASE
        WHEN l.loan_status = 'npl'     THEN 1
        WHEN l.loan_status = 'healthy' THEN 0
    END AS ust,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 1)  AS fht1,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 2)  AS fht2,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 3)  AS fht3,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 4)  AS fht4,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 5)  AS fht5,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 6)  AS fht6,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 7)  AS fht7,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'fht' AND (elem->>'questionNumber')::int = 8)  AS fht8,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 11) AS kmsi1,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 12) AS kmsi2,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 13) AS kmsi3,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 14) AS kmsi4,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 15) AS kmsi5,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 16) AS kmsi6,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 17) AS kmsi7,
    (SELECT (elem->>'choiceNumber')::int FROM jsonb_array_elements(a.answer) AS elem
     WHERE elem->>'group' = 'kmsi' AND (elem->>'questionNumber')::int = 18) AS kmsi8
FROM stg_users u
LEFT JOIN stg_answers      a ON a.user_id    = u.user_id
LEFT JOIN stg_features     f ON f.feature_id = a.feature_id
LEFT JOIN stg_loan_summary l ON l.user_id    = u.user_id
WHERE f.feature_slug = 'fsk_v2.0'
  AND a.answer IS NOT NULL
  AND (
      (l.loan_status = 'npl'     AND l.payoff_score < 0)
   OR (l.loan_status = 'healthy' AND l.payoff_score >= 5)
  );


-- ─────────────────────────────────────────────
-- 3. int_loan_features_mapped
--    Score scale inversions/coalescing.
--    Mirrors dbt/intermediate/int_loan_features_mapped.sql
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW int_loan_features_mapped AS
SELECT
    user_id,
    ust,
    CASE fht1 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht1,
    CASE fht2 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht2,
    CASE fht3 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht3,
    CASE fht4 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht4,
    CASE fht5 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht5,
    CASE fht6 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht6,
    CASE fht7 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht7,
    CASE fht8 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS fht8,
    COALESCE(kmsi1, 0) AS kmsi1,
    COALESCE(kmsi2, 0) AS kmsi2,
    COALESCE(kmsi3, 0) AS kmsi3,
    COALESCE(kmsi4, 0) AS kmsi4,
    COALESCE(kmsi5, 0) AS kmsi5,
    COALESCE(kmsi6, 0) AS kmsi6,
    CASE kmsi7 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS kmsi7,
    CASE kmsi8 WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END AS kmsi8
FROM stg_loan_features_extracted;


-- ─────────────────────────────────────────────
-- 4. int_loan_features_engineered
--    Group stats + cross-group ratios + interactions.
--    Mirrors dbt/intermediate/int_loan_features_engineered.sql
--    (calculate_group_stats macro expanded inline)
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW int_loan_features_engineered AS
WITH group_stats AS (
    SELECT
        user_id, ust,
        (fht1 + fht2) AS spending_score_sum,
        ((fht1 + fht2) / 2) AS spending_score_avg,
        (CASE WHEN fht1=3 THEN 1 ELSE 0 END + CASE WHEN fht2=3 THEN 1 ELSE 0 END) AS spending_high_score_count,
        (fht3 + fht4) AS saving_score_sum,
        ((fht3 + fht4) / 2) AS saving_score_avg,
        (CASE WHEN fht3=3 THEN 1 ELSE 0 END + CASE WHEN fht4=3 THEN 1 ELSE 0 END) AS saving_high_score_count,
        (fht5 + fht6) AS payoff_score_sum,
        ((fht5 + fht6) / 2) AS payoff_score_avg,
        (CASE WHEN fht5=3 THEN 1 ELSE 0 END + CASE WHEN fht6=3 THEN 1 ELSE 0 END) AS payoff_high_score_count,
        (fht7 + fht8) AS planning_score_sum,
        ((fht7 + fht8) / 2) AS planning_score_avg,
        (CASE WHEN fht7=3 THEN 1 ELSE 0 END + CASE WHEN fht8=3 THEN 1 ELSE 0 END) AS planning_high_score_count,
        (kmsi1 + kmsi2) AS loan_score_sum,
        ((kmsi1 + kmsi2) / 2) AS loan_score_avg,
        (CASE WHEN kmsi1=3 THEN 1 ELSE 0 END + CASE WHEN kmsi2=3 THEN 1 ELSE 0 END) AS loan_high_score_count,
        (kmsi3 + kmsi4) AS worship_score_sum,
        ((kmsi3 + kmsi4) / 2) AS worship_score_avg,
        (CASE WHEN kmsi3=3 THEN 1 ELSE 0 END + CASE WHEN kmsi4=3 THEN 1 ELSE 0 END) AS worship_high_score_count,
        (kmsi5 + kmsi6) AS extravagance_score_sum,
        ((kmsi5 + kmsi6) / 2) AS extravagance_score_avg,
        (CASE WHEN kmsi5=3 THEN 1 ELSE 0 END + CASE WHEN kmsi6=3 THEN 1 ELSE 0 END) AS extravagance_high_score_count,
        (kmsi7 + kmsi8) AS vigilance_score_sum,
        ((kmsi7 + kmsi8) / 2) AS vigilance_score_avg,
        (CASE WHEN kmsi7=3 THEN 1 ELSE 0 END + CASE WHEN kmsi8=3 THEN 1 ELSE 0 END) AS vigilance_high_score_count
    FROM int_loan_features_mapped
)
SELECT
    *,
    (loan_score_sum::FLOAT         / (saving_score_sum      + 1)) AS loan_to_saving_ratio,
    (worship_score_sum::FLOAT      / (vigilance_score_sum   + 1)) AS worship_to_vigilance_ratio,
    (extravagance_score_sum::FLOAT / (spending_score_sum    + 1)) AS extravagance_to_spending_ratio,
    (worship_score_sum::FLOAT      / (payoff_score_sum      + 1)) AS worship_to_payoff_ratio,
    (loan_score_avg        * extravagance_score_avg) AS loan_extravagance_interaction,
    (payoff_score_avg      * planning_score_avg)     AS payoff_planning_interaction,
    (spending_score_avg    * vigilance_score_avg)    AS spending_vigilance_interaction,
    (worship_score_avg     * extravagance_score_avg) AS worship_extravagance_interaction
FROM group_stats;


-- ─────────────────────────────────────────────
-- 5. fct_loan_features  (what Dagster reads)
--    Mirrors dbt/marts/fct_loan_features.sql
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW fct_loan_features AS
    SELECT * FROM int_loan_features_engineered;

-- =============================================================================
-- END V001
-- =============================================================================
