with mapped as (
    select * from {{ ref('int_loan_features_mapped') }}
),

group_stats as (
    select
        user_id,
        ust,
        {{ calculate_group_stats('spending', ['fht1', 'fht2']) }},
        {{ calculate_group_stats('saving', ['fht3', 'fht4']) }},
        {{ calculate_group_stats('payoff', ['fht5', 'fht6']) }},
        {{ calculate_group_stats('planning', ['fht7', 'fht8']) }},
        {{ calculate_group_stats('loan', ['kmsi1', 'kmsi2']) }},
        {{ calculate_group_stats('worship', ['kmsi3', 'kmsi4']) }},
        {{ calculate_group_stats('extravagance', ['kmsi5', 'kmsi6']) }},
        {{ calculate_group_stats('vigilance', ['kmsi7', 'kmsi8']) }}
    from mapped
)

select
    *,
    -- Ratios (Sum / (Denominator + 1) to avoid division by zero)
    (loan_score_sum::float / (saving_score_sum + 1)) as loan_to_saving_ratio,
    (worship_score_sum::float / (vigilance_score_sum + 1)) as worship_to_vigilance_ratio,
    (extravagance_score_sum::float / (spending_score_sum + 1)) as extravagance_to_spending_ratio,
    (worship_score_sum::float / (payoff_score_sum + 1)) as worship_to_payoff_ratio,
    
    -- Interactions (Avg * Avg)
    (loan_score_avg * extravagance_score_avg) as loan_extravagance_interaction,
    (payoff_score_avg * planning_score_avg) as payoff_planning_interaction,
    (spending_score_avg * vigilance_score_avg) as spending_vigilance_interaction,
    (worship_score_avg * extravagance_score_avg) as worship_extravagance_interaction
from group_stats
