select
    user_id,
    feature_id,
    answer
from {{ source('raw_data', 'answers') }}
