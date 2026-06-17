"""
prepare_data.py

Two modes:

  python prepare_data.py --demo
      Creates a tiny fake dataset (20 rows) so you can test the
      full pipeline without waiting for the real data to download.

  python prepare_data.py --convert
      Converts the real CrisisMMD agreed-label .tsv files into
      the train/dev/test CSV format expected by dataset.py

  python prepare_data.py --check
      Verifies your CSV files and image counts are correct.
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image

DATA_DIR      = "data/crisismmd"
IMAGE_DIR     = os.path.join(DATA_DIR, "images")
SPLIT_DIR     = DATA_DIR

SPLIT_FILES = {
    "train": "task_humanitarian_text_img_agreed_lab_train.tsv",
    "dev":   "task_humanitarian_text_img_agreed_lab_dev.tsv",
    "test":  "task_humanitarian_text_img_agreed_lab_test.tsv",
}

LABEL_MAP = {
    "infrastructure_and_utility_damage":      "infrastructure_damage",
    "infrastructure and utility damage":      "infrastructure_damage",
    "affected_individuals":                   "affected_individuals",
    "affected individuals":                   "affected_individuals",
    "injured_or_dead_people":                 "affected_individuals",
    "injured or dead people":                 "affected_individuals",
    "missing_or_found_people":                "affected_individuals",
    "missing or found people":                "affected_individuals",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "rescue volunteering or donation effort": "rescue_volunteering_or_donation_effort",
    "vehicle_damage":                         "infrastructure_damage",
    "vehicle damage":                         "infrastructure_damage",
    "other_relevant_information":             "other_relevant_information",
    "other relevant information":             "other_relevant_information",
    "not_humanitarian":                       "not_humanitarian",
    "not humanitarian":                       "not_humanitarian",
    "not_relevant_or_cant_judge":             "not_humanitarian",
    "not relevant or can't judge":            "not_humanitarian",
}

VALID_LABELS = {
    "infrastructure_damage",
    "affected_individuals",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
}

DEMO_TWEETS = [
    ("Hurricane Maria damage: 100,000 homes without power after storm",       "infrastructure_damage",                   "3 4"),
    ("#RedCross assisting thousands impacted by devastating wildfires",        "rescue_volunteering_or_donation_effort",  "1 2 3"),
    ("4 killed in Cyclone Mora, 140 houses destroyed in Mizoram",             "affected_individuals",                    "0 1 2 3"),
    ("All tornado and severe thunderstorm warnings for August",               "other_relevant_information",              "1 2"),
    ("Beautiful sunset over the mountains today",                             "not_humanitarian",                        ""),
    ("Storm Harvey flood victims face displaced alligators",                  "affected_individuals",                    "2 3 4"),
    ("Fundraiser for hurricane Irma and Harvey victims",                      "rescue_volunteering_or_donation_effort",  "0 1 2"),
    ("Flooded cars could flood used car market after Harvey",                 "infrastructure_damage",                   "0 1"),
    ("Emergency supplies being distributed at local shelters",                "rescue_volunteering_or_donation_effort",  "0 1 2 3"),
    ("Earthquake magnitude 7.1 strikes central Mexico",                       "infrastructure_damage",                   "0 1 2"),
    ("Missing persons hotline activated for flood victims",                   "affected_individuals",                    "0 1 2"),
    ("Government declares state of emergency after floods",                   "other_relevant_information",              "0 1 2 3"),
    ("Volunteers needed at community center for relief efforts",              "rescue_volunteering_or_donation_effort",  "0"),
    ("Category 5 hurricane makes landfall in Puerto Rico",                    "infrastructure_damage",                   "0 1 2"),
    ("Residents urged to evacuate as wildfire spreads",                       "other_relevant_information",              "0 1 2 3"),
    ("Rescue teams searching for survivors in collapsed building",            "rescue_volunteering_or_donation_effort",  "0 1 2 3"),
    ("Family of five missing after flash flood in Texas",                     "affected_individuals",                    "0 1 2 3 4"),
    ("Power outages affect 2 million after hurricane strikes",                "infrastructure_damage",                   "0 1 2"),
    ("Local bakery donates 500 meals to disaster relief center",              "rescue_volunteering_or_donation_effort",  "2 3 4 5 6"),
    ("Weather service issues tornado watch for three counties",               "other_relevant_information",              "0 1 2 3"),
]


def create_demo_dataset():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    colors = [(200,100,100),(100,200,100),(100,100,200),(200,200,100),(200,100,200)]
    rows = []
    for i, (text, label, rationale) in enumerate(DEMO_TWEETS):
        img_name = f"demo_{i:03d}.jpg"
        Image.new("RGB", (200, 200), color=colors[i % len(colors)]).save(
            os.path.join(IMAGE_DIR, img_name))
        rows.append({"text": text, "image_path": img_name,
                     "label": label, "rationales": rationale})

    df    = pd.DataFrame(rows)
    n     = len(df)
    t_end = int(n * 0.70)
    d_end = int(n * 0.85)
    df.iloc[:t_end].to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    df.iloc[t_end:d_end].to_csv(os.path.join(DATA_DIR, "dev.csv"),   index=False)
    df.iloc[d_end:].to_csv(os.path.join(DATA_DIR, "test.csv"),  index=False)
    print(f"Demo dataset created in {DATA_DIR}/")
    print(f"  train.csv : {t_end} rows")
    print(f"  dev.csv   : {d_end - t_end} rows")
    print(f"  test.csv  : {n - d_end} rows")
    print(f"  images/   : {n} placeholder images")
    print("\nRun the app:  python app.py")


def find_tsv_file(split):
    candidates = [
        os.path.join(SPLIT_DIR, SPLIT_FILES[split]),
        os.path.join(SPLIT_DIR, "crisismmd_datasplit_agreed_label", SPLIT_FILES[split]),
        os.path.join(SPLIT_DIR, "data", SPLIT_FILES[split]),
        os.path.join("CrisisMMD_v2.0", "CrisisMMD_v2.0", "annotations", SPLIT_FILES[split]),
        os.path.join("crisismmd_datasplit_agreed_label", "crisismmd_datasplit_agreed_label", SPLIT_FILES[split]),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def convert_crisismmd():
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for split in ["train", "dev", "test"]:
        tsv_path = find_tsv_file(split)
        if tsv_path is None:
            print(f"\nCould not find {split} TSV. Expected:")
            print(f"  {os.path.join(SPLIT_DIR, SPLIT_FILES[split])}")
            print(f"Make sure you extracted crisismmd_datasplit_agreed_label.zip into {DATA_DIR}/")
            continue

        print(f"\nReading {tsv_path} ...")
        df = pd.read_csv(tsv_path, sep="\t", on_bad_lines="skip")
        print(f"  Shape   : {df.shape}")
        print(f"  Columns : {list(df.columns)}")

        cols_lower = {c.lower(): c for c in df.columns}
        col_map = {}
        for key, variations in {
            "text":       ["tweet_text", "text", "tweet text"],
            "image_path": ["image", "image_id", "image_path", "image_file"],
            "label":      ["label", "class_label", "humanitarian_label"],
        }.items():
            for v in variations:
                if v in cols_lower:
                    col_map[key] = cols_lower[v]
                    break

        missing = [k for k in ["text", "image_path", "label"] if k not in col_map]
        if missing:
            print(f"  Cannot find columns for: {missing}")
            print(f"  Available: {list(df.columns)}")
            continue

        df_out = pd.DataFrame()
        df_out["text"]       = df[col_map["text"]].fillna("").astype(str)
        df_out["image_path"] = df[col_map["image_path"]].fillna("").astype(str)
        df_out["label"]      = (df[col_map["label"]].fillna("").astype(str)
                                 .str.strip().str.lower()
                                 .map(lambda x: LABEL_MAP.get(x, x)))
        df_out["rationales"] = ""

        before = len(df_out)
        df_out = df_out[df_out["label"].isin(VALID_LABELS)]
        print(f"  Rows after label filter: {len(df_out)} (dropped {before - len(df_out)})")
        print(f"  Class distribution:")
        for lbl, cnt in df_out["label"].value_counts().items():
            print(f"    {lbl}: {cnt}")

        out_path = os.path.join(DATA_DIR, f"{split}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

    print("\nConversion complete! Next: python app.py")


def check_dataset():
    print("\n── Dataset check ─────────────────────────────────────")
    for split in ["train", "dev", "test"]:
        path = os.path.join(DATA_DIR, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n{split}.csv  ({len(df)} rows)")
            if "label" in df.columns:
                for lbl, cnt in df["label"].value_counts().items():
                    print(f"  {lbl:50s} {cnt}")
            if "image_path" in df.columns and len(df) > 0:
                img_name = df["image_path"].iloc[0]
                found = (os.path.exists(os.path.join(DATA_DIR, img_name)) or
                         os.path.exists(os.path.join(IMAGE_DIR, img_name)))
                print(f"  First image ({img_name}): {'OK' if found else 'NOT FOUND'}")
        else:
            print(f"\n{split}.csv → NOT FOUND")

    img_count = sum(
        len([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])
        for _, _, files in os.walk(DATA_DIR)
    )
    print(f"\nTotal images under {DATA_DIR}: {img_count}")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--check",   action="store_true")
    args = parser.parse_args()

    if args.demo:
        create_demo_dataset()
    elif args.convert:
        convert_crisismmd()
    elif args.check:
        check_dataset()
    else:
        print("Usage:")
        print("  python prepare_data.py --demo      # fake data for quick testing")
        print("  python prepare_data.py --convert   # convert real CrisisMMD files")
        print("  python prepare_data.py --check     # verify data setup")