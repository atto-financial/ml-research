-- SQL for loan data loading
-- This file picks up data from the dbt-managed model
SELECT
    *
FROM
    fct_loan_features
