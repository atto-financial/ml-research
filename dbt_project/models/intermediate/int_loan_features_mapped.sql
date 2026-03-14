with extracted as (
    select * from {{ ref('stg_loan_features_extracted') }}
)

select
    user_id,
    ust,
    -- Mapping fht (Spending, Saving, Payoff, Planning): 1->3, 2->2, 3->1
    case fht1 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht1,
    case fht2 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht2,
    case fht3 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht3,
    case fht4 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht4,
    case fht5 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht5,
    case fht6 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht6,
    case fht7 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht7,
    case fht8 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as fht8,
    
    -- Mapping kmsi1-6 (Loan, Worship, Extravagance): 1->1, 2->2, 3->3
    coalesce(kmsi1, 0) as kmsi1,
    coalesce(kmsi2, 0) as kmsi2,
    coalesce(kmsi3, 0) as kmsi3,
    coalesce(kmsi4, 0) as kmsi4,
    coalesce(kmsi5, 0) as kmsi5,
    coalesce(kmsi6, 0) as kmsi6,
    
    -- Mapping kmsi7-8 (Vigilance): 1->3, 2->2, 3->1
    case kmsi7 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as kmsi7,
    case kmsi8 when 1 then 3 when 2 then 2 when 3 then 1 else 0 end as kmsi8
from extracted
