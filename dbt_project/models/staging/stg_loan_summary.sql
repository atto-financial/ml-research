select
    user_id,
    loan_status,
    payoff_score
from {{ source('raw_data', 'loan_summary_statuses') }}
