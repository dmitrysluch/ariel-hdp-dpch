#!/bin/bash
set -eo pipefail

SESSION='{ "table": "analytics_data",  "user_id": 1, "queries_total": 10, "queries_left": 1000, "max_eps": 1, "max_delta": 0.000001, "eps_used": 0, "delta_used": 0, "noise_type": "gaussian"}'

# QUERY='{"tp": "covariance", "column1": {"tp": "column", "name": "value1", "min_val": 0, "max_val": 12, "size": 5}, "column2": {"tp": "column", "name": "value2", "min_val": 0, "max_val": 25, "size": 5}}'

QUERY='{"tp": "mean", "column": {"tp": "column", "name": "value1", "min_val": 0, "max_val": 12, "size": 5}}'

# curl -XGET 'http://localhost:8000/schema' -H "Content-Type: application/json" -d "$SESSION" | jq .

curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -d "{ \"session\": $SESSION, \"query\": $QUERY }"