#!/usr/bin/env python3
"""
Select 50 diagnostic reports from the 4,349 inference pool covering known difficult boundaries.

Selection strategy:
- Fixed random seed for reproducibility
- Deterministic keyword/semantic boundary selection
- Covers all 15 known difficult boundary categories
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

from experiments.ragset_report_inference_experiment.src.ragset_inference.data import load_train, split_gold, LABEL_COLUMNS

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Boundary categories with keyword patterns (case-insensitive)
BOUNDARY_PATTERNS = {
    "meniscal_degeneration_vs_tear": [
        r"degenerat", r"degenerative", r"mucoid", r"myxoid", r"intrasubstance",
        r"grade\s*[12]", r"signal.*not.*tear", r"no.*tear", r"without.*tear",
        r"discoid", r"meniscosis", r"meniscal.*signal.*no.*contact"
    ],
    "suspected_possible_ro_meniscal_tear": [
        r"suspect", r"possible", r"probable", r"likely", r"cannot\s*exclude",
        r"rule\s*out", r"r/o", r"questionable", r"equivocal", r"indeterminate",
        r"may\s*represent", r"suggestive\s*of", r"consistent\s*with.*tear"
    ],
    "acl_mcl_injury_vs_degeneration_sprain": [
        r"sprain", r"partial.*tear", r"high.signal", r"increased.*signal",
        r"thickened", r"lax", r"attenuated", r"mucoid.*degeneration",
        r"ganglion", r"cyst.*acl", r"cyst.*mcl", r"insertion.*edema",
        r"enthesopathy", r"mucoid.*change"
    ],
    "compartment_specific_oa": [
        r"medial.*compartment.*oa", r"lateral.*compartment.*oa",
        r"patellofemoral.*oa", r"pf.*oa", r"medial.*oa", r"lateral.*oa",
        r"femorotibial.*medial", r"femorotibial.*lateral",
        r"isolated.*medial", r"isolated.*lateral"
    ],
    "generalized_tricompartmental_oa": [
        r"tricompartmental", r"generalized.*oa", r"diffuse.*oa",
        r"global.*oa", r"pancompartmental", r"three.*compartment",
        r"all.*compartment", r"widespread.*oa"
    ],
    "chondromalacia_chondropathy": [
        r"chondromalacia", r"chondropathy", r"chondrosis", r"chondral.*injury",
        r"chondral.*defect", r"fissuring", r"fibrillation", r"grade\s*[34]",
        r"full.thickness", r"exposed.*bone", r"retropatellar.*chondro"
    ],
    "effusion_vs_synovitis": [
        r"effusion", r"joint.*fluid", r"fluid.*joint", r"suprapatellar",
        r"bursa", r"synovitis", r"synovial.*thickening", r"synovial.*hyperplasia",
        r"synovial.*enhancement", r"pannus"
    ],
    "explicit_synovitis": [
        r"synovitis", r"synovial.*thickening", r"synovial.*hyperplasia",
        r"synovial.*enhancement", r"pannus", r"villous", r"nodular.*synov"
    ],
    "bakers_cyst": [
        r"baker", r"popliteal.*cyst", r"gastrocnemius.*semimembranosus",
        r"medial.*gastrocnemius", r"semimembranosus.*bursa"
    ],
    "bone_marrow_edema_vs_contusion": [
        r"bone.*marrow.*edema", r"marrow.*edema", r"bme", r"edema.*bone",
        r"contusion", r"bone.*bruise", r"impaction", r"subchondral.*edema",
        r"transient.*osteoporosis", r"osteonecrosis", r"avascular.*necrosis"
    ],
    "fracture_vs_osteochondral_injury": [
        r"fracture", r"fx", r"break", r"cortical.*break", r"cortical.*discontinuity",
        r"osteochondral", r"ocd", r"osteochondritis", r"loose.*body",
        r"avulsion", r"tibial.*spine", r"intercondylar.*eminence"
    ],
    "historical_vs_current_findings": [
        r"history\s*of", r"previous", r"prior", r"old", r"chronic",
        r"post.?op", r"postoperative", r"status.?post", r"s/p",
        r"previous.*surgery", r"prior.*injury", r"old.*tear", r"chronic.*tear",
        r"healed", r"post.traumatic"
    ],
    "explicit_negative_findings": [
        r"normal", r"intact", r"no.*evidence", r"no.*sign", r"absent",
        r"without.*abnormality", r"unremarkable", r"negative\s*for",
        r"no.*tear", r"no.*fracture", r"no.*edema", r"no.*effusion"
    ],
    "multilingual_reports": [
        # Non-English patterns
        r"técnica|resultados|impresión|hallazgos|constataciones|bevindingen|klinische",
        r"fractures?|alignement|changements|chondropathie|œdème|médullaire",
        r"мр|находка|костномозъчен|едем|кондил|хрущял|руптура|предната|кръстна",
        r"mr|bulgular|menisküs|bağ|tendon|patella|retinaküler|devamsızlı",
        r"rotura|menisco|condilo|artrosis|derrame|necrosis|avascular|subcondral"
    ],
    "contradictory_statements": [
        r"but.*no", r"however.*no", r"although.*no", r"despite.*no",
        r"except.*no", r"without.*tear", r"signal.*but.*intact",
        r"edema.*but.*no.*fracture", r"fluid.*but.*no.*synovitis"
    ],
}

def categorize_report(report_text: str) -> list[str]:
    """Return list of boundary categories this report matches."""
    text_lower = report_text.lower()
    categories = []
    for category, patterns in BOUNDARY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                categories.append(category)
                break
    return categories

def select_diagnostic_50():
    """Select 50 diagnostic reports covering all boundary categories."""
    df = load_train('train.csv')
    gold, unlabeled = split_gold(df)
    
    print(f"Total unlabeled reports: {len(unlabeled)}")
    
    # Categorize all unlabeled reports
    report_categories = defaultdict(list)
    for idx, row in unlabeled.iterrows():
        categories = categorize_report(row['Report'])
        for cat in categories:
            report_categories[cat].append((idx, row))
    
    # Print category coverage
    print("\nCategory coverage in unlabeled pool:")
    for cat, reports in report_categories.items():
        print(f"  {cat}: {len(reports)} reports")
    
    # Select reports ensuring coverage of all categories
    selected = []
    selected_indices = set()
    
    # Priority order: categories with fewer matches first (harder to cover)
    category_priority = sorted(report_categories.keys(), key=lambda c: len(report_categories[c]))
    
    # First pass: ensure at least 2 reports per category (where available)
    for cat in category_priority:
        available = [(idx, row) for idx, row in report_categories[cat] if idx not in selected_indices]
        if available:
            # Take up to 2 per category
            n_take = min(2, len(available))
            chosen = np.random.choice(len(available), size=n_take, replace=False)
            for i in chosen:
                idx, row = available[i]
                selected.append((idx, row, cat))
                selected_indices.add(idx)
    
    print(f"\nAfter category coverage pass: {len(selected)} reports selected")
    
    # Second pass: fill remaining slots with diverse reports
    remaining_slots = 50 - len(selected)
    if remaining_slots > 0:
        # Get all unselected reports
        all_unselected = [(idx, row) for idx, row in unlabeled.iterrows() if idx not in selected_indices]
        # Prefer reports that match multiple categories
        scored = []
        for idx, row in all_unselected:
            cats = categorize_report(row['Report'])
            score = len(cats)
            scored.append((score, idx, row, cats))
        
        # Sort by score descending, then random
        scored.sort(key=lambda x: (-x[0], np.random.random()))
        
        for score, idx, row, cats in scored[:remaining_slots]:
            primary_cat = cats[0] if cats else "other"
            selected.append((idx, row, primary_cat))
            selected_indices.add(idx)
    
    # If still not enough, just take random
    if len(selected) < 50:
        all_unselected = [(idx, row) for idx, row in unlabeled.iterrows() if idx not in selected_indices]
        needed = 50 - len(selected)
        chosen = np.random.choice(len(all_unselected), size=min(needed, len(all_unselected)), replace=False)
        for i in chosen:
            idx, row = all_unselected[i]
            cats = categorize_report(row['Report'])
            primary_cat = cats[0] if cats else "other"
            selected.append((idx, row, primary_cat))
    
    # Trim to exactly 50
    selected = selected[:50]
    
    # Create output DataFrame
    output_rows = []
    for idx, row, primary_cat in selected:
        all_cats = categorize_report(row['Report'])
        output_rows.append({
            'StudyInstanceUID': row['StudyInstanceUID'],
            'Report': row['Report'],
            'primary_boundary_category': primary_cat,
            'all_boundary_categories': ';'.join(all_cats) if all_cats else 'none',
            'selection_seed': SEED,
        })
    
    output_df = pd.DataFrame(output_rows)
    
    # Save selection
    output_path = Path('experiments/ragset_report_inference_experiment/data/validation/diagnostic_50.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"\nSelected {len(output_df)} reports saved to {output_path}")
    print("\nSelection summary:")
    print(output_df['primary_boundary_category'].value_counts())
    
    # Show first few
    print("\nFirst 5 selected reports:")
    for _, row in output_df.head().iterrows():
        print(f"  {row['StudyInstanceUID']}: {row['primary_boundary_category']} - {row['Report'][:100]}...")
    
    return output_df

if __name__ == "__main__":
    select_diagnostic_50()