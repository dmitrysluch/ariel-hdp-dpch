#!/bin/bash
set -eo pipefail

SESSION='{"session": { "dataset": "vius1", "queries_total": 1000, "queries_left": 1000, "max_eps": 0.5, "max_delta": 0.00001, "eps_used": 0, "delta_used": 0, "noise_type": "laplacian"}}'

AUTH=$(curl -XPOST 'http://localhost:8000/auth/new' -H "Content-Type: application/json" -d "$SESSION" | jq .token -r)
echo $AUTH

curl -XGET 'http://localhost:8000/schema' -H "Content-Type: application/json" -H "Authorization: Bearer $AUTH" --silent | jq .

QUERY='{"tp": "sum", "columns": {"tp":"columns","dataframe": "vius1", "columns":["avgweight", "regstate_AZ"]}}'
curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -H "Authorization: Bearer $AUTH" -d "{ \"query\": $QUERY }" | jq .

QUERY='{"tp": "mean", "columns": {"tp":"columns","dataframe": "vius1", "columns":["acquireyear"]}}'
curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -H "Authorization: Bearer $AUTH" -d "{ \"query\": $QUERY }" | jq .

QUERY='{"tp": "covariance", "columns1": {"tp":"columns","dataframe": "vius1", "columns":["acquireyear"]}}'
curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -H "Authorization: Bearer $AUTH" -d "{ \"query\": $QUERY }" | jq .

QUERY='{"tp": "histogram", "bins": 10, "columns": {"tp":"columns","dataframe": "vius1", "columns":["avgweight"]}}'
curl -XPOST 'http://localhost:8000/query' -H "Content-Type: application/json" -H "Authorization: Bearer $AUTH" -d "{ \"query\": $QUERY }" | jq .