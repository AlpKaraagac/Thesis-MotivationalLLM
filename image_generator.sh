#!/usr/bin/env bash

set -euo pipefail

mkdir -p data/images

for json_file in data/extractions_discoverllm/*.json; do
    artifact_id="$(basename "$json_file" .json)"

    echo "Processing ${artifact_id}"

    #
    # regrouped extraction
    #
    python scripts/visualize_intent_tree.py \
        "$json_file" \
        --out "data/images/${artifact_id}/regrouped.png"

    #
    # original extraction
    #
    original_json="data/extractions/${artifact_id}.json"

    if [[ -f "$original_json" ]]; then
        python scripts/visualize_intent_tree.py \
            "$original_json" \
            --out "data/images/${artifact_id}/original.png"
    else
        echo "Warning: missing ${original_json}"
    fi

    #
    # discoverllm tree (red nodes hidden)
    #
    python scripts/visualize_discoverllm_tree.py \
        "$artifact_id" \
        --out "data/images/${artifact_id}/discoverllm_tree"

    python scripts/visualize_discoverllm_tree.py \
        "$artifact_id" \
        --out "data/images/${artifact_id}/discoverllm_tree_red" \
        --show-red

done

echo "Done."