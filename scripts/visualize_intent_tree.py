#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from graphviz import Digraph


def visualize_tree(tree, output_dir):
    tree_id = tree["tree_id"]

    dot = Digraph(
        name=tree_id,
        format="png",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
        }
    )

    dot.attr("node", shape="box")

    nodes = {n["id"]: n for n in tree["nodes"]}

    for node in tree["nodes"]:
        label = (
            f"{node['label']}\n"
            f"(turn {node['turn_of_discovery']})"
        )

        dot.node(
            node["id"],
            label=label
        )

    for node in tree["nodes"]:
        parent = node.get("parent_id")
        if parent:
            dot.edge(parent, node["id"])

    outfile = output_dir / f"{tree_id}"
    dot.render(str(outfile), cleanup=True)


def visualize_extraction(path, output_dir=None):
    with open(path) as f:
        doc = json.load(f)

    if output_dir is None:
        output_dir = Path("tree_viz") / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for tree in doc["trees"]:
        visualize_tree(tree, output_dir)

    print(f"saved -> {output_dir}")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        help="Path to the JSON file containing the tree extraction"
    )

    parser.add_argument(
        "--out",
        help="output directory"
    )

    args = parser.parse_args()

    visualize_extraction(
        path=Path(args.file),
        output_dir=Path(args.out) if args.out else None
    )
