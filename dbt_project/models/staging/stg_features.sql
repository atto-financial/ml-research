select
    feature_id,
    feature_slug
from {{ source('raw_data', 'features') }}
