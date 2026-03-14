-- This fact model now serves as the final interface for the engineered features,
-- pulling from the intermediate layers where the logic resides.

select
    *
from {{ ref('int_loan_features_engineered') }}
