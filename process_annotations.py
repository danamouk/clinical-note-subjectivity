#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import argparse

def extract_text_spans_and_label(df, padding_left=50, padding_right=50):
    """
    Extract context phrase, word, and label from text spans.
    """
    df["PHRASE"] = [
        df.loc[i, "TEXT"][max(0, df.loc[i, "BEGIN_WORD"] - padding_left) : df.loc[i, "END_WORD"] + padding_right]
        for i in range(len(df))
    ]
    df["WORD"] = [df.loc[i, "TEXT"][df.loc[i, "BEGIN_WORD"] : df.loc[i, "END_WORD"]] for i in range(len(df))]
    df["LABEL"] = [0 if x == "rejected" else 1 for x in df["ANNOTATION"]]
    return df

def main():
    parser = argparse.ArgumentParser(description="Process MIMIC-III note annotations and extract phrases.")
    parser.add_argument("--notes", required=True, help="Path to NOTEEVENTS.csv file")
    parser.add_argument("--annotations", required=True, help="Path to annotations.csv file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--padding_left", type=int, default=50, help="Padding to the left of phrase (default: 50)")
    parser.add_argument("--padding_right", type=int, default=50, help="Padding to the right of phrase (default: 50)")

    args = parser.parse_args()

    # Load notes data
    notes = pd.read_csv(args.notes)
    print(f"Total notes: {len(notes)}")
    print(f"Note columns: {notes.columns.tolist()}")

    # Load annotations
    annotation_df = pd.read_csv(args.annotations)
    annotation_df["row_id"] = annotation_df["row_id"].astype(str)
    notes["ROW_ID"] = notes["ROW_ID"].astype(str)

    # Merge
    merged_df = notes.merge(annotation_df, left_on="ROW_ID", right_on="row_id", how="inner")
    print(f"Merged dataset length: {len(merged_df)}")

    # Count accepted/rejected
    accepted_count = merged_df["annotation"].str.count("accepted").sum()
    rejected_count = merged_df["annotation"].str.count("rejected").sum()
    total_count = accepted_count + rejected_count
    print(f"Accepted: {accepted_count}, Rejected: {rejected_count}, Total: {total_count}")
    print(f"Merged columns: {merged_df.columns.tolist()}")

    # Uppercase columns
    merged_df.columns = merged_df.columns.str.upper()

    # Rename annotation column for consistency
    merged_df.rename(columns={"ANNOTATION": "ANNOTATION", "BEGIN": "BEGIN_WORD", "END": "END_WORD"}, inplace=True)

    # Extract phrases and labels
    merged_df = extract_text_spans_and_label(merged_df, padding_left=args.padding_left, padding_right=args.padding_right)

    # Order columns
    ordered_columns = [
        "ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "CATEGORY",
        "TEXT", "BEGIN_WORD", "END_WORD", "WORD", "PHRASE", "LABEL"
    ]
    merged_df = merged_df[ordered_columns]
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Clean HADM_ID
    merged_df["HADM_ID"] = pd.to_numeric(merged_df["HADM_ID"], errors="coerce").astype("Int64")

    # Save without TEXT column
    output_df = merged_df.drop(columns="TEXT")
    output_df.to_csv(args.output, index=False)
    print(f"Saved cleaned file to: {args.output}")

    # Label counts
    count_label_0 = (output_df["LABEL"] == 0).sum()
    count_label_1 = (output_df["LABEL"] == 1).sum()
    print(f"Number of 0s in LABEL: {count_label_0}")
    print(f"Number of 1s in LABEL: {count_label_1}")

    # Explore categories
    categories_0 = output_df[output_df["LABEL"] == 0]["CATEGORY"].unique()
    categories_1 = output_df[output_df["LABEL"] == 1]["CATEGORY"].unique()
    print(f"Categories in LABEL 0: {categories_0}")
    print(f"Categories in LABEL 1: {categories_1}")

    # Category counts
    categories_in_label_0 = output_df[output_df["LABEL"] == 0]["CATEGORY"].value_counts()
    categories_in_label_1 = output_df[output_df["LABEL"] == 1]["CATEGORY"].value_counts()
    print(f"Category counts in LABEL 0:\n{categories_in_label_0}")
    print(f"Category counts in LABEL 1:\n{categories_in_label_1}")

if __name__ == "__main__":
    main()

# Usage:
# python process_annotations.py --notes NOTEEVENTS.csv --annotations annotations.csv --output mimiciii_implicit_bias_notes.csv
