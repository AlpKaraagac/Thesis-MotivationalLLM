#!/usr/bin/env python3

"""
Visualize a DiscoverLLM intent tree for a specific artifact.

Usage:

    python visualize_tree.py artifact_1
    python visualize_tree.py artifact_1 --show-red

By default:
    - Green = satisfied > 0
    - Yellow = aware > 0 and satisfied == 0
    - Red nodes are omitted

With --show-red:
    - Green = satisfied > 0
    - Yellow = aware > 0 and satisfied == 0
    - Red = aware == 0 and satisfied == 0
"""

import ast
import json
import argparse

import pandas as pd
from graphviz import Digraph


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "data/discoverllm_creative_writing_conversations.csv"


# ============================================================
# TREE FILTERING
# ============================================================

def has_visible_descendant(node):
    """
    Used when red nodes are hidden.

    Keep:
      - green nodes
      - yellow nodes
      - red nodes that are needed to connect
        to a visible descendant
    """
    aware = node.get("aware", 0)
    satisfied = node.get("satisfied", 0)

    if aware > 0 or satisfied > 0:
        return True

    return any(
        has_visible_descendant(child)
        for child in node.get("children", [])
    )


def should_render(node, show_red):
    if show_red:
        return True

    return has_visible_descendant(node)


# ============================================================
# GRAPHVIZ
# ============================================================

def node_color(node, show_red):
    aware = node.get("aware", 0)
    satisfied = node.get("satisfied", 0)

    if satisfied > 0:
        return "#c8f7c5"  # green

    if aware > 0:
        return "#fff3b0"  # yellow

    if show_red:
        return "#f5c2c7"  # red

    return "#eeeeee"      # structural connector


def add_node(dot, node, show_red):
    if not should_render(node, show_red):
        return False

    aware = node.get("aware", 0)
    satisfied = node.get("satisfied", 0)

    label = node["text"]

    # Optional metadata
    label += (
        f"\n\n"
        f"aware={aware} "
        f"satisfied={satisfied}"
    )

    reason = node.get("reason", "")

    if reason:
        reason = reason.replace("\n", " ")

        if len(reason) > 120:
            reason = reason[:117] + "..."

        label += f"\n\n{reason}"

    dot.node(
        node["id"],
        label,
        style="filled",
        fillcolor=node_color(node, show_red),
        shape="box",
    )

    for child in node.get("children", []):
        rendered = add_node(dot, child, show_red)

        if rendered:
            dot.edge(node["id"], child["id"])

    return True


# ============================================================
# CSV LOADING
# ============================================================

def load_trees(csv_path, artifact_id):
    df = pd.read_csv(csv_path)

    matches = df[df["artifact_id"] == artifact_id]

    if len(matches) == 0:
        raise ValueError(
            f"Artifact '{artifact_id}' not found in {csv_path}"
        )

    row = matches.iloc[0]

    # Change this column name if needed.
    tree_col = "trees"

    raw = row[tree_col]
    try:                                   # JSON (new helper output)
        trees = json.loads(raw)
    except Exception:                      # Python repr (legacy CSVs)
        trees = ast.literal_eval(raw)

    return trees


# ============================================================
# MAIN
# ============================================================

def visualize_artifact(artifact_id, show_red, out_path=None):
    trees = load_trees(CSV_PATH, artifact_id)

    dot = Digraph(
        name=artifact_id,
        format="png",
    )

    dot.attr(
        rankdir="LR",
        splines="ortho",
    )

    title = artifact_id

    if not show_red:
        title += " (red nodes hidden)"

    dot.attr(
        label=title,
        labelloc="t",
        fontsize="20",
    )

    for root in trees:
        add_node(dot, root, show_red)

    if out_path is None:
        out_path = f"tree_viz/{artifact_id}"
    dot.render(out_path, cleanup=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "artifact_id",
        help="Artifact ID to visualize"
    )

    parser.add_argument(
        "--show-red",
        action="store_true",
        help="Include red (unaware/unsatisfied) nodes"
    )

    parser.add_argument(
        "--out",
        help="Output path"
    )


    args = parser.parse_args()

    visualize_artifact(
        artifact_id=args.artifact_id,
        show_red=args.show_red,
        out_path=args.out
    )


if __name__ == "__main__":
    main()