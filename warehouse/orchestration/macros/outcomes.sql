{#
    PA outcome taxonomy.
    Source of truth: xg/outcomes.yaml
    Sync validated by: tests/test_outcome_sync.py
#}

{% macro outcome_classes() %}
    {{ return([
        'double', 'double_play', 'field_out',
        'fielders_choice', 'fielders_choice_out', 'force_out',
        'grounded_into_double_play', 'hit_by_pitch', 'home_run',
        'sac_bunt', 'sac_fly', 'single', 'strikeout',
        'strikeout_double_play', 'triple', 'walk'
    ]) }}
{% endmacro %}

{% macro excluded_outcomes() %}
    {{ return([
        'catcher_interf', 'ejection', 'field_error',
        'game_advisory', 'intent_walk', 'sac_bunt_double_play',
        'sac_fly_double_play', 'triple_play', 'truncated_pa'
    ]) }}
{% endmacro %}

{% macro excluded_tuple() %}
    ({% for o in excluded_outcomes() %}'{{ o }}'{% if not loop.last %}, {% endif %}{% endfor %})
{% endmacro %}
