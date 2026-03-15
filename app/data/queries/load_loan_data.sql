-- SQL for loan data loading
-- This file picks up data from the dbt-managed model

-- Filter: only users assigned to the 'student' loan group (ST001)
-- Joined via loan_summary_statuses which has direct user_id + loan_group_id

SELECT
    f.*
FROM
    fct_loan_features f
INNER JOIN loan_summary_statuses lss ON lss.user_id = f.user_id
INNER JOIN loan_groups lg ON lg.loan_group_id = lss.loan_group_id
WHERE
    lg.code = 'ST001'
