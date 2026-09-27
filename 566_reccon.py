import csv
import json

TRAIN_CSV = "/Users/sai-work/Documents/gitCode/new/orchestrate-agent-lab/train.csv"

JSONL_IN = (
    "/Users/sai-work/Documents/gitCode/new/orchestrate-agent-lab/"
    "experiments/ragset_report_inference_experiment/results/inference/"
    "needs_review_56_reports.jsonl"
)

JSONL_OUT = (
    "/Users/sai-work/Documents/gitCode/new/orchestrate-agent-lab/"
    "experiments/ragset_report_inference_experiment/results/inference/"
    "needs_review_56_reports_populated.jsonl"
)

# ---------------------------------------------------------
# 1. Read train.csv and index by StudyInstanceUID
# ---------------------------------------------------------

train_by_id = {}

with open(TRAIN_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    required = {
        "StudyInstanceUID",
        "Report",
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    }

    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing train.csv columns: {sorted(missing)}")

    for row in reader:
        uid = row["StudyInstanceUID"].strip()

        if uid:
            train_by_id[uid] = row


# ---------------------------------------------------------
# 2. Read the 56 needs_review records
# ---------------------------------------------------------

records = []

with open(JSONL_IN, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            records.append(json.loads(line))


# ---------------------------------------------------------
# 3. Match report_id -> StudyInstanceUID
#    and populate original_report
# ---------------------------------------------------------

matched = 0
missing = []

for record in records:
    report_id = str(record["report_id"]).strip()

    train_row = train_by_id.get(report_id)

    if train_row is None:
        missing.append(report_id)
        continue

    record["original_report"] = train_row["Report"]
    matched += 1


# ---------------------------------------------------------
# 4. Safety checks
# ---------------------------------------------------------

assert len(records) == 56, (
    f"Expected 56 records, found {len(records)}"
)

if missing:
    print("\nWARNING: These IDs were not found in train.csv:")
    for uid in missing:
        print(uid)

print(f"\nTotal needs_review records : {len(records)}")
print(f"Matched to train.csv       : {matched}")
print(f"Missing from train.csv     : {len(missing)}")


# ---------------------------------------------------------
# 5. Write populated JSONL
# ---------------------------------------------------------

with open(JSONL_OUT, "w", encoding="utf-8") as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

print(f"\nWritten:")
print(JSONL_OUT)