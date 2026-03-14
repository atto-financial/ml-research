with answers as (
    select * from {{ ref('stg_answers') }}
),
features as (
    select * from {{ ref('stg_features') }}
),
loan_summary as (
    select * from {{ ref('stg_loan_summary') }}
),
users as (
    select * from {{ ref('stg_users') }}
)

select
    u.user_id,
    case
        when l.loan_status = 'npl' then 1
        when l.loan_status = 'healthy' then 0
    end as ust,
    -- Extracting fht features
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 1) as fht1,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 2) as fht2,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 3) as fht3,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 4) as fht4,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 5) as fht5,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 6) as fht6,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 7) as fht7,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'fht' and (elem->>'questionNumber')::int = 8) as fht8,
    -- Extracting kmsi features
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 11) as kmsi1,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 12) as kmsi2,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 13) as kmsi3,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 14) as kmsi4,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 15) as kmsi5,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 16) as kmsi6,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 17) as kmsi7,
    (select (elem->>'choiceNumber')::int from jsonb_array_elements(a.answer::jsonb) as elem where elem->>'group' = 'kmsi' and (elem->>'questionNumber')::int = 18) as kmsi8
from
    users u
left join answers a on a.user_id = u.user_id
left join features f on f.feature_id = a.feature_id
left join loan_summary l on l.user_id = u.user_id
where f.feature_slug = 'fsk_v2.0'
    and a.answer is not null
    -- Filter for training data (as per fct_loan_features)
    and ((l.loan_status = 'npl' and l.payoff_score < 0) or (l.loan_status = 'healthy' and l.payoff_score >= 5))
