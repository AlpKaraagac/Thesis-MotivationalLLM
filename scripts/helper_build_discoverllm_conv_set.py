# Pull the data from https://huggingface.co/datasets/kixlab/DiscoverLLM-multiturn-preferences
# Per each artifact_id, take max turn_id and max_score (just one item there can be max 2 in this case take the first)
# For each of these data points, conversation is prompt + completion.
# prompt column is a list of dict with role and content, we can convert it to a conversation format with role as speaker and content as text.
# Completion is a string and the role is always assistant
# And we can extract the intent tree from `criteria_history → hierarchy` (nested `{id, text, children}`).

"""
Example hierarchy:
"hierarchy": [{"id": "1", "text": "Conversation between narrator and a historical figure", "children": [{"id": "1.1", "text": "Dialogue-based narrative format", "children": [{"id": "1.1.1", "text": "Entirely dialogue-based narrative format", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.5658919834043828, 0.27663385037741284]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.03205585009331169, 0.9592274535397637]}, {"id": "1.2", "text": "Conversation between narrator and Napoleon Bonaparte", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.14670826750180344, 0.7819220897346079]}, {"id": "1.3", "text": "Historical figure complains about their portrayal", "children": [{"id": "1.3.1", "text": "Napoleon complains about his portrayal", "children": [{"id": "1.3.1.1", "text": "Napoleon appears to complain about his portrayal in a previous fictional story", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.6372132632324382, 0.22647750435149494]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.8033876653182412, 0.17695881577207806]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.830890798470021, 0.02815419141748421]}, {"id": "1.4", "text": "Napoleon's speech has distinctive speech pattern or accent", "children": [{"id": "1.4.1", "text": "Napoleon's speech rendered in exaggerated phonetic French accent", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.4976065776446762, 0.28077428549906824]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.8390794515755546, 0.3706124998002214]}, {"id": "1.5", "text": "Distinguished formatting for Napoleon's speech versus narrator's speech", "children": [{"id": "1.5.1", "text": "Uses italics to distinguish Napoleon's speech from narrator's speech", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.5081436151312889, 0.6807054806657803]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.7100597396788375, 0.596438912716726]}, {"id": "1.6", "text": "Maintains humorous, self-aware tone throughout", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.6331793478404218, 0.7315018245852098]}, {"id": "1.7", "text": "Includes meta-fictional elements about the writing process", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.041086843769185144, 0.716020989789378]}, {"id": "1.8", "text": "References contemporary cultural references", "children": [{"id": "1.8.1", "text": "References contemporary internet culture", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.9658770890899094, 0.7798473567469121]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.9576665806539457, 0.8633002955715329]}, {"id": "1.9", "text": "Introduces another historical figure as second complainant", "children": [{"id": "1.9.1", "text": "Introduces Genghis Khan as second complainant", "children": [{"id": "1.9.1.1", "text": "Concludes with cliffhanger introducing Genghis Khan as second complainant", "children": [], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.8560082348489418, 0.45897825920338653]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.737654127262324, 0.9515304449115084]}], "aware": 0, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": [0.09814207028606858, 0.9374554838694857]}], "aware": 1, "satisfied": 0, "reason": "", "is_update": false, "update_thresholds": []

"""
# We can then save these conversations in a csv file with columns: artifact_id, config, conversation (list of dict with speaker and text), tree (nested dict with id, text, children).
# This can be used for training and evaluating intent tree extraction models.

import os
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

def iter_nested(node):
    yield node
    for c in node.get("children", []):
        yield from iter_nested(c)

def nested_depth(node):
    ch = node.get("children", [])
    return 1 if not ch else 1 + max(nested_depth(c) for c in ch)

def nested_branching(root):
    internal = [n for n in iter_nested(root) if n.get("children")]
    return round(sum(len(n["children"]) for n in internal) / len(internal), 2) if internal else 0.0

def extract_hierarchies(s):
    roots = []
    try:
        parsed = json.loads(s) if isinstance(s, str) else s
    except Exception:
        return roots
    stack, seen = [parsed], set()
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            if isinstance(x.get("hierarchy"), list):
                k = str(x.get("criterion", ""))[:80]
                if k not in seen:
                    seen.add(k)
                    roots.extend(x["hierarchy"])
            for v in x.values():
                if isinstance(v, (list, dict)):
                    stack.append(v)
    return roots

if __name__ == "__main__":
    DATASET = "kixlab/DiscoverLLM-multiturn-preferences"
    CONFIGS = ["creative_writing", "svg_drawing", "technical_writing"]
    
    for cfg in CONFIGS:
        ds = load_dataset(DATASET, cfg, split="train", token=os.getenv("HF_TOKEN"))

        df_raw = ds.to_pandas()

        df_raw = df_raw.sort_values(
            ["artifact_id", "turn_id", "score"],
            ascending=[True, False, False]
        )

        df_raw = df_raw.drop_duplicates(
            subset=["artifact_id"],
            keep="first"
        )

        rows = []
        
        for ex in df_raw.to_dict("records"):
            artifact_id = ex["artifact_id"]

            conversation = [
                {"speaker": turn["role"], "text": turn["content"]}
                for turn in ex["prompt"]
            ]
            conversation.append({
                "speaker": "assistant",
                "text": ex["completion"]
            })

            trees = extract_hierarchies(ex.get("criteria_history", []))

            rows.append({
                "artifact_id": artifact_id,
                "config": cfg,
                "conversation": conversation,
                "trees": trees,
            })
        df = pd.DataFrame(rows)
        # Serialize nested columns as JSON (robust quoting/escaping; readers json.loads them).
        df["conversation"] = df["conversation"].apply(json.dumps)
        df["trees"] = df["trees"].apply(json.dumps)
        out_path = Path(f"data/discoverllm_{cfg}_conversations.csv") 
        df.to_csv(out_path, index=False) 
        print(f"Saved {len(df)} conversations with trees to {out_path}")
            

