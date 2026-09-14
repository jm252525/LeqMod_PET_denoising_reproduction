#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    missing = []
    checked_rows = 0
    for name in ("train.csv", "val.csv", "test_bern.csv", "test_ruijin.csv"):
        with (input_dir / name).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                checked_rows += 1
                for field in ("low_count_path", "full_count_path"):
                    if not Path(row[field]).is_file():
                        missing.append({
                            "manifest": name,
                            "site": row["site"],
                            "patient_id": row["patient_id"],
                            "study_id": row["study_id"],
                            "count_label": row["count_label"],
                            "field": field,
                            "path": row[field],
                        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("manifest", "site", "patient_id", "study_id", "count_label", "field", "path")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)
    print(f"ROWS={checked_rows} PATHS={checked_rows * 2} MISSING={len(missing)}")
    print("MISSING_BY_MANIFEST=" + repr(dict(Counter(row["manifest"] for row in missing))))
    print(f"REPORT={output}")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
