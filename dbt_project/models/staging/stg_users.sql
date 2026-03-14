select
    user_id
from {{ source('raw_data', 'users') }}
