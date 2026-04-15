"""Integrity checks for intent trees and forests."""

from __future__ import annotations

from discoverllm.types import IntentForest, IntentNode, IntentTree
from discoverllm.validation.schema import validate_node_id


def _validate_parent_child(parent: IntentNode, child: IntentNode) -> None:
    expected_prefix = f"{parent.id}."
    if not child.id.startswith(expected_prefix):
        raise ValueError(
            f"Child id {child.id!r} must start with parent prefix {expected_prefix!r}."
        )
    if child.id.count(".") != parent.id.count(".") + 1:
        raise ValueError(
            f"Child id {child.id!r} is not an immediate descendant of parent {parent.id!r}."
        )


def validate_tree(tree: IntentTree) -> None:
    if not isinstance(tree, IntentTree):
        raise TypeError("tree must be an IntentTree.")
    if tree.root.id.count(".") != 0:
        raise ValueError("Tree root id must be a top-level id such as '1'.")

    seen_ids: set[str] = set()

    def visit(node: IntentNode) -> None:
        normalized = validate_node_id(node.id)
        if normalized in seen_ids:
            raise ValueError(f"Duplicate node id {normalized!r} in tree.")
        seen_ids.add(normalized)

        sibling_ids: set[str] = set()
        for child in node.children:
            _validate_parent_child(node, child)
            if child.id in sibling_ids:
                raise ValueError(
                    f"Duplicate child id {child.id!r} under parent {node.id!r}."
                )
            sibling_ids.add(child.id)
            visit(child)

    visit(tree.root)


def validate_forest(forest: IntentForest) -> None:
    if not isinstance(forest, IntentForest):
        raise TypeError("forest must be an IntentForest.")
    if not forest.trees:
        raise ValueError("IntentForest must contain at least one tree.")

    root_ids: set[str] = set()
    for tree in forest.trees:
        validate_tree(tree)
        root_id = tree.root.id
        if root_id in root_ids:
            raise ValueError(f"Duplicate root id {root_id!r} in forest.")
        root_ids.add(root_id)
