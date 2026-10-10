{#
  dbt's default prefixes the custom schema onto the profile's — `+schema: processed`
  against profile schema `processed` yields `processed_processed`, not `processed`.
  Override so the declared schema is the schema, and models land where named.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
