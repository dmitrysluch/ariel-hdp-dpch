#!/bin/bash
set -eo pipefail

SESSION='{ "dataset": "vius1", "queries_total": 1000, "queries_left": 1000, "max_eps": 0.5, "max_delta": 0.00001, "eps_used": 0, "delta_used": 0, "noise_type": "laplacian"}'

AUTH=$(curl -XPOST 'http://localhost:8000/auth/new' -H "Content-Type: application/json" -d "$SESSION")

# QUERY='{"tp": "covariance", "column1": {"tp":"column","name":"acquireyear","min_val":1998.0,"max_val":2021.0,"size":67952}, "column2": {"tp":"column","name":"acquireyear","min_val":1998.0,"max_val":2021.0,"size":67952}}'

# # QUERY='{"tp": "mean", "column": {"tp":"column","name":"acquireyear","min_val":1998.0,"max_val":2021.0,"size":67952}}'

curl -XGET 'http://localhost:8000/schema' -H "Content-Type: application/json" -d "$SESSION" --silent | jq .

# curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -d "{ \"session\": $SESSION, \"query\": $QUERY }"