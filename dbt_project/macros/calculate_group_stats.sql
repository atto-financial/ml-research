{% macro calculate_group_stats(group_name, columns) %}
    -- Sum of scores
    ({% for col in columns %}
        {{ col }}{% if not loop.last %} + {% endif %}
    {% endfor %}) as {{ group_name }}_score_sum,
    
    -- Average of scores
    (({% for col in columns %}
        {{ col }}{% if not loop.last %} + {% endif %}
    {% endfor %}) / {{ columns|length }}) as {{ group_name }}_score_avg,

    -- High score count (number of questions answered with 3)
    ({% for col in columns %}
        case when {{ col }} = 3 then 1 else 0 end{% if not loop.last %} + {% endif %}
    {% endfor %}) as {{ group_name }}_high_score_count
{% endmacro %}
