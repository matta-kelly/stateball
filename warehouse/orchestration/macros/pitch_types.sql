{#
    Canonical pitch type list for per-pitch-type arsenal features.
    Used by: int_mlb__pitcher_arsenal_counts, int_mlb__pitcher_arsenal, feat_mlb__vectors.
    Python mirror: training/config.py::PITCH_TYPES
    Adding a new pitch type: add here, update PITCH_TYPES in config.py,
    run --full-refresh on int_mlb__pitcher_arsenal_counts.
#}

{% macro pitch_types() %}
    {{ return(['FF', 'SI', 'FC', 'SL', 'CU', 'KC', 'SV', 'ST', 'CH', 'FS']) }}
{% endmacro %}
