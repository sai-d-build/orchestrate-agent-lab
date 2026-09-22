#!/usr/bin/env python3
"""LLM-based gold analysis for RagSet.

Runs once to analyze the 58 gold reports and persists structured analysis
to results/gold/gold_analysis.json. This replaces the basic statistical
analysis with rich LLM-discovered patterns.

Features:
- 3-4 labels per LLM call per report to avoid token limits and gateway timeouts
- Retry with exponential backoff for 504/503/5002/5003 errors
- Checkpointing for manual restart capability
- 100s wait for API overload errors
- Robust JSON parsing with truncation handling
"""

from pathlib import Path
import json
import yaml
import os
import logging
import sys
import time
import random
import re
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gold_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

from experiments.ragset_report_inference_experiment.src.ragset_inference.data import (
    load_train, split_gold, LABEL_COLUMNS,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import (
    InferenceModel, _is_api_overload_error, _get_retry_delay
)

# Checkpoint file path
CHECKPOINT_FILE = Path("gold_analysis_checkpoint.json")

# Label batch size: 1-2 labels per call to keep output under ~8K tokens
# Reduced from 3-4 to prevent JSON truncation
LABEL_BATCH_SIZE = 2


def load_gold_analysis_prompt(exp_root: Path) -> dict:
    """Load the gold analysis prompt template."""
    prompt_path = exp_root / "prompts" / "gold_analysis.yaml"
    return yaml.safe_load(prompt_path.read_text(encoding="utf-8"))


def format_single_report_for_analysis(row, labels_subset) -> str:
    """Format a single gold report for the LLM analysis prompt.

    Includes specified labels for the report so the LLM can analyze
    cross-label patterns, co-occurrences, and contradictions within one report.

    Args:
        row: Single row from gold DataFrame
        labels_subset: List of label names for this batch
    """
    labels = {label: int(row[label]) for label in labels_subset}
    return (
        f"STUDY {row['StudyInstanceUID']}\n"
        f"LABELS: {labels}\n"
        f"REPORT:\n{row['Report']}"
    )


def build_output_structure(labels_subset) -> str:
    """Build the expected JSON output structure for the given labels.
    
    Compact structure to reduce token usage and prevent truncation.
    """
    label_specific_template = ",\n".join([f'    "{label}": {{}}' for label in labels_subset])

    return f"""{{
  "metadata": {{"total_gold_reports": 1, "labels_analyzed": {len(labels_subset)}, "labels_in_batch": {json.dumps(labels_subset)}}},
  "label_specific": {{
{label_specific_template}
  }},
  "linguistic": {{"negation_patterns": {{}}, "uncertainty_patterns": {{}}, "historical_current_patterns": {{}}, "question_indication_patterns": {{}}, "abbreviations": {{}}, "multilingual_terminology": {{}}}},
  "clinical": {{"anatomy": {{}}, "laterality": {{}}, "related_labels": {{}}, "confusing_findings": [], "difficult_cases": []}},
  "contradictions": [],
  "derived_conventions": [],
  "unsafe_shortcuts": [],
  "unresolved_rules": [],
  "quality_control": {{}}
}}"""


def run_gold_analysis_batch(model: InferenceModel, prompt_cfg: dict, gold_df, labels_subset, report_idx) -> dict:
    """Run LLM-based gold analysis on a single report for a subset of labels."""
    row = gold_df.iloc[report_idx]
    study_id = row['StudyInstanceUID']
    
    logger.info(f"Formatting report {report_idx + 1}/{len(gold_df)} (Study {study_id}) for labels: {labels_subset}...")
    gold_examples = format_single_report_for_analysis(row, labels_subset)

    labels_text = "\n".join(labels_subset)

    logger.debug(f"Gold examples length: {len(gold_examples)} characters")
    logger.debug(f"Labels: {labels_text}")

    # Strengthen the user prompt with explicit field requirements (compact version)
    field_requirements = f"""
MANDATORY: Your response MUST include ALL top-level fields:
- metadata, label_specific (keys: {", ".join(labels_subset)}), linguistic, clinical, contradictions, derived_conventions, unsafe_shortcuts, unresolved_rules, quality_control

For EACH label in label_specific, provide:
- gold_positive_count (0 or 1), gold_negative_count (0 or 1)
- raw_observations (array with study_instance_uid, exact_text, normalized_pattern, evidence_strength, gold_label, anatomical_scope, certainty, temporal_status, context, related_findings)
- pattern_frequencies (array with pattern, total_occurrences, gold_positive, gold_negative, positive_rate, unique_studies, sample_size)
- contradictions, safe_rules, unsafe_shortcuts, unresolved_rules, anatomy_specifics

Output COMPLETE JSON only. No extra text."""

    user = prompt_cfg["user_template"].format(
        labels=labels_text,
        gold_examples=gold_examples,
    ) + field_requirements
    
    logger.debug(f"User prompt length: {len(user)} characters")

    output_structure = build_output_structure(labels_subset)

    logger.info(f"Calling LLM model: {model.model} with max_tokens: {model.max_tokens}")
    logger.debug(f"System prompt length: {len(prompt_cfg['system'])} characters")

    # Retry with exponential backoff for API overload errors (502, 503, 504, 5002, 5003, etc.)
    # For API overload errors, wait 100 seconds; for others, use exponential backoff: 5, 15, 30 seconds
    max_retries = 3
    retry_delays = [5, 15, 30]  # seconds
    overload_wait = 100.0  # seconds for API overload errors

    for attempt in range(max_retries):
        try:
            response = model.client.chat.completions.create(
                model=model.model,
                messages=[
                    {"role": "system", "content": prompt_cfg["system"] + "\n\nCRITICAL: You MUST output ONLY valid JSON. No explanatory text, no markdown, no reasoning. The response must be a single JSON object matching the structure specified below."},
                    {"role": "user", "content": user + "\n\nOUTPUT REQUIREMENT: Return ONLY a valid JSON object matching this exact structure. No other text:\n" + output_structure},
                ],
                response_format={"type": "json_object"},
                max_tokens=model.max_tokens,
                temperature=0,
            )
            logger.info("LLM response received successfully")
            break
        except Exception as e:
            is_overload = _is_api_overload_error(e)
            is_retryable = is_overload or attempt < max_retries - 1

            if is_retryable and attempt < max_retries - 1:
                if is_overload:
                    wait_time = overload_wait
                    logger.warning(f"API overload detected (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                else:
                    wait_time = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    logger.warning(f"Retryable error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")

                time.sleep(wait_time)
                continue
            else:
                logger.error(f"LLM call failed: {e}")
                raise
    else:
        raise RuntimeError(f"Max retries ({max_retries}) exceeded for LLM call")

    text = response.choices[0].message.content
    if text is None:
        logger.warning("Response content is None, using empty string")
        text = ""
    logger.debug(f"Response text length: {len(text)} characters")
    logger.debug(f"Response text preview: {text[:500]}")

    # Clean up common JSON formatting issues from LLM
    text = text.strip()
    # Fix double opening brace issue (including with newlines/spaces)
    text = re.sub(r'^\{\s*\{', '{', text)
    # Fix if response starts with ```json
    if text.startswith('```json'):
        text = text[7:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()

    # Try to fix truncated JSON by closing open structures
    def try_fix_truncated_json(text):
        """Attempt to fix common truncation issues in JSON."""
        # First, try to find the last complete object
        # Look for the last complete key-value pair
        text = text.strip()
        
        # Count braces and brackets
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        # If truncated in the middle of a string, try to close it
        # Count quotes to detect unclosed strings
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if ch == '\\' and not escaped:
                escaped = True
            elif ch == '"' and not escaped:
                in_string = not in_string
            else:
                escaped = False
        
        if in_string:
            # Find the last quote and truncate there, then close
            last_quote = text.rfind('"')
            if last_quote > 0:
                text = text[:last_quote + 1]
        
        # Re-count after potential string fix
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
        return text

    try:
        result = json.loads(text)
        logger.info("Successfully parsed JSON response")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        # Try to fix truncated JSON
        fixed_text = try_fix_truncated_json(text)
        try:
            result = json.loads(fixed_text)
            logger.info("Successfully parsed fixed JSON response")
            return result
        except json.JSONDecodeError as e2:
            logger.warning(f"Failed to parse fixed JSON: {e2}")

        # Try to find valid JSON within the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                logger.info("Successfully extracted and parsed JSON from response")
                return result
            except json.JSONDecodeError:
                pass
        # Fallback: return basic structure with raw text
        return {
            "raw_analysis": text,
            "metadata": {},
            "dataset_level": {},
            "label_specific": {label: {} for label in labels_subset},
            "linguistic": {},
            "clinical": {},
            "contradictions": [],
            "derived_conventions": [],
            "unsafe_shortcuts": [],
            "unresolved_rules": [],
            "quality_control": {}
        }


def load_checkpoint() -> dict:
    """Load checkpoint from file."""
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {"completed_batches": [], "analysis": {}}


def save_checkpoint(completed_batches: list, analysis: dict):
    """Save checkpoint to file."""
    try:
        checkpoint = {
            "completed_batches": completed_batches,
            "analysis": analysis,
            "timestamp": time.time()
        }
        with CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {len(completed_batches)} batches completed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def validate_batch_analysis(analysis: dict, labels_subset) -> bool:
    """Validate that analysis contains meaningful data for the batch labels."""
    if not analysis:
        return False
    if "raw_analysis" in analysis and not analysis.get("label_specific"):
        return False

    label_specific = analysis.get("label_specific", {})
    if not label_specific:
        return False

    # Check that all batch labels have data
    for label in labels_subset:
        if label not in label_specific:
            logger.warning(f"Missing label in analysis: {label}")
            return False
        label_data = label_specific[label]
        if not label_data:
            logger.warning(f"Label {label} has empty data")
            return False
        # Allow zero counts if at least one is present (some labels may be all positive or all negative in this report)
        if label_data.get("gold_positive_count", 0) == 0 and label_data.get("gold_negative_count", 0) == 0:
            logger.warning(f"Label {label} has zero counts")
            return False

    return True


def merge_analyses(accumulated: dict, batch_result: dict, labels_subset) -> dict:
    """Merge batch result into accumulated analysis."""
    if not accumulated:
        accumulated = {
            "metadata": batch_result.get("metadata", {}),
            "dataset_level": batch_result.get("dataset_level", {}),
            "label_specific": {},
            "linguistic": batch_result.get("linguistic", {}),
            "clinical": batch_result.get("clinical", {}),
            "contradictions": batch_result.get("contradictions", []),
            "derived_conventions": batch_result.get("derived_conventions", []),
            "unsafe_shortcuts": batch_result.get("unsafe_shortcuts", []),
            "unresolved_rules": batch_result.get("unresolved_rules", []),
            "quality_control": batch_result.get("quality_control", {})
        }

    # Merge label_specific
    for label in labels_subset:
        if label in batch_result.get("label_specific", {}):
            # For single-report batches, we need to accumulate observations
            if label not in accumulated["label_specific"]:
                accumulated["label_specific"][label] = batch_result["label_specific"][label]
            else:
                # Merge observations and pattern frequencies
                existing = accumulated["label_specific"][label]
                new = batch_result["label_specific"][label]
                
                # Merge raw_observations
                if "raw_observations" in new and new["raw_observations"]:
                    if "raw_observations" not in existing:
                        existing["raw_observations"] = []
                    existing["raw_observations"].extend(new["raw_observations"])
                
                # Merge pattern_frequencies
                if "pattern_frequencies" in new and new["pattern_frequencies"]:
                    if "pattern_frequencies" not in existing:
                        existing["pattern_frequencies"] = []
                    existing["pattern_frequencies"].extend(new["pattern_frequencies"])
                
                # Update counts
                existing["gold_positive_count"] = existing.get("gold_positive_count", 0) + new.get("gold_positive_count", 0)
                existing["gold_negative_count"] = existing.get("gold_negative_count", 0) + new.get("gold_negative_count", 0)
                
                # Merge other arrays
                for key in ["contradictions", "safe_rules", "unsafe_shortcuts", "unresolved_rules"]:
                    if key in new and new[key]:
                        if key not in existing:
                            existing[key] = []
                        existing[key].extend(new[key])
                
                # Merge anatomy_specifics
                if "anatomy_specifics" in new and new["anatomy_specifics"]:
                    if "anatomy_specifics" not in existing:
                        existing["anatomy_specifics"] = {}
                    existing["anatomy_specifics"].update(new["anatomy_specifics"])

    # Merge other sections (extend lists, update dicts)
    for key in ["linguistic", "clinical"]:
        if key in batch_result and batch_result[key]:
            if key not in accumulated:
                accumulated[key] = {}
            # Deep merge for dicts
            if isinstance(batch_result[key], dict):
                for subkey, value in batch_result[key].items():
                    if subkey not in accumulated[key]:
                        accumulated[key][subkey] = value
                    elif isinstance(value, dict) and isinstance(accumulated[key][subkey], dict):
                        accumulated[key][subkey].update(value)
                    elif isinstance(value, list) and isinstance(accumulated[key][subkey], list):
                        accumulated[key][subkey].extend(value)

    for key in ["contradictions", "derived_conventions", "unsafe_shortcuts", "unresolved_rules"]:
        if key in batch_result and batch_result[key]:
            if key not in accumulated:
                accumulated[key] = []
            accumulated[key].extend(batch_result[key])

    if "quality_control" in batch_result and batch_result["quality_control"]:
        if "quality_control" not in accumulated:
            accumulated["quality_control"] = {}
        accumulated["quality_control"].update(batch_result["quality_control"])

    return accumulated


def main():
    logger.info("Starting gold analysis (3-4 labels per batch per report)...")
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    logger.debug(f"Experiment root: {exp}")

    # Load config
    logger.info("Loading experiment config...")
    cfg = yaml.safe_load((exp / "config/experiment.yaml").read_text(encoding="utf-8"))
    logger.debug(f"Config: {cfg}")

    # Load gold reports
    logger.info("Loading gold reports...")
    df = load_train(root / cfg["source"]["path"])
    gold, heldout = split_gold(df)
    logger.info(f"Loaded {len(gold)} gold reports, {len(heldout)} heldout reports")

    # Load prompt
    logger.info("Loading gold analysis prompt...")
    prompt_cfg = load_gold_analysis_prompt(exp)
    logger.debug(f"Prompt keys: {prompt_cfg.keys()}")

    # Load model profiles for fallback chain
    logger.info("Loading model config...")
    models_cfg = yaml.safe_load((root / "config" / "models.yaml").read_text(encoding="utf-8"))
    
    # Define fallback chain for gold analysis
    fallback_profiles = [
        cfg["models"].get("gold_analysis_profile"),
        cfg["models"].get("gold_analysis_profile", "").replace("-usf", "") + "-usf" if "usf" not in cfg["models"].get("gold_analysis_profile", "") else None,
        "ragset-gold-analysis",  # OpenRouter fallback
    ]
    # Filter out None and duplicates
    fallback_profiles = [p for p in fallback_profiles if p and p in models_cfg.get("models", {})]
    # Remove duplicates while preserving order
    seen = set()
    fallback_profiles = [p for p in fallback_profiles if not (p in seen or seen.add(p))]
    
    logger.info(f"Gold analysis fallback chain: {fallback_profiles}")
    
    model = None
    gold_config = None
    gold_profile = None
    
    for profile_name in fallback_profiles:
        if profile_name not in models_cfg.get("models", {}):
            logger.warning(f"Profile {profile_name} not found in models config, skipping")
            continue
            
        gold_config = models_cfg["models"][profile_name]
        gold_provider = gold_config.get("provider", "nvidia")
        gold_api_key_env = gold_config.get("api_key_env", "NVIDIA_API_KEY")
        
        if not os.environ.get(gold_api_key_env):
            logger.warning(f"API key {gold_api_key_env} not set for profile {profile_name}, skipping")
            continue
            
        logger.info(f"Trying gold analysis profile: {profile_name}")
        logger.info(f"Model: {gold_config['model']}, Provider: {gold_provider}")
        
        try:
            model = InferenceModel(
                gold_config["model"],
                max_tokens=gold_config.get("parameters", {}).get("max_output_tokens", 16000),
                provider=gold_provider,
                api_key=os.environ.get(gold_api_key_env),
                reasoning=False,
            )
            logger.info(f"Model initialized: {model.model} with max_tokens={model.max_tokens}")
            gold_profile = profile_name
            break
        except Exception as e:
            logger.warning(f"Failed to initialize model for profile {profile_name}: {e}")
            continue
    
    if model is None:
        raise RuntimeError("Failed to initialize any gold analysis model from fallback chain")
    
    logger.info(f"Successfully initialized gold analysis model: {gold_profile}")

    # Load checkpoint for resume capability
    checkpoint = load_checkpoint()
    completed_batches = checkpoint.get("completed_batches", [])
    accumulated_analysis = checkpoint.get("analysis", {})

    # Create label batches (3-4 labels per batch)
    label_batches = [LABEL_COLUMNS[i:i + LABEL_BATCH_SIZE] for i in range(0, len(LABEL_COLUMNS), LABEL_BATCH_SIZE)]
    num_reports = len(gold)
    total_batches = num_reports * len(label_batches)
    logger.info(f"Processing {num_reports} reports x {len(label_batches)} label batches = {total_batches} total batches")

    batch_num = 0
    for report_idx in range(num_reports):
        row = gold.iloc[report_idx]
        study_id = row['StudyInstanceUID']
        
        for label_batch_idx, labels_subset in enumerate(label_batches):
            batch_num += 1
            batch_key = f"report_{report_idx}_{study_id}_labels_{label_batch_idx}_{'_'.join(labels_subset)}"

            if batch_key in completed_batches:
                logger.info(f"Skipping completed batch: {batch_key}")
                continue

            logger.info(f"Processing batch {batch_num}/{total_batches}: {batch_key}")

            # Retry logic for batch processing with exponential backoff
            max_batch_retries = 3
            batch_retry_delays = [5, 15, 30]  # seconds
            batch_result = None
            
            for batch_attempt in range(max_batch_retries):
                try:
                    # Single call with one report for 1-2 labels
                    batch_result = run_gold_analysis_batch(model, prompt_cfg, gold, labels_subset, report_idx)
                    
                    # Validate the batch analysis
                    if not validate_batch_analysis(batch_result, labels_subset):
                        logger.warning(f"Batch validation failed for {batch_key} (attempt {batch_attempt + 1}/{max_batch_retries})")
                        if batch_attempt < max_batch_retries - 1:
                            wait_time = batch_retry_delays[batch_attempt] if batch_attempt < len(batch_retry_delays) else batch_retry_delays[-1]
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Batch validation failed for {batch_key} after {max_batch_retries} attempts")
                            # Save checkpoint with partial results but don't mark complete
                            save_checkpoint(completed_batches, accumulated_analysis)
                            raise RuntimeError(f"Gold analysis validation failed for batch {batch_key} after {max_batch_retries} attempts. Check logs for details.")
                    
                    # Success
                    break
                    
                except Exception as e:
                    logger.warning(f"Batch attempt {batch_attempt + 1}/{max_batch_retries} failed: {e}")
                    if batch_attempt < max_batch_retries - 1:
                        wait_time = batch_retry_delays[batch_attempt] if batch_attempt < len(batch_retry_delays) else batch_retry_delays[-1]
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Batch failed after {max_batch_retries} attempts: {e}")
                        save_checkpoint(completed_batches, accumulated_analysis)
                        raise

            # Merge batch result into accumulated analysis
            accumulated_analysis = merge_analyses(accumulated_analysis, batch_result, labels_subset)

            # Mark batch as completed
            completed_batches.append(batch_key)
            save_checkpoint(completed_batches, accumulated_analysis)

    # Final validation: all 12 labels should have data
    logger.info("Validating complete analysis...")
    all_labels_present = all(label in accumulated_analysis.get("label_specific", {}) for label in LABEL_COLUMNS)
    all_labels_have_data = all(
        accumulated_analysis.get("label_specific", {}).get(label, {}).get("gold_positive_count", 0) > 0 or
        accumulated_analysis.get("label_specific", {}).get(label, {}).get("gold_negative_count", 0) > 0
        for label in LABEL_COLUMNS
    )

    if not all_labels_present or not all_labels_have_data:
        logger.error("Final validation failed - not all labels have data")
        save_checkpoint(completed_batches, accumulated_analysis)
        raise RuntimeError("Gold analysis incomplete. Check logs for details.")

    # Update metadata to reflect total reports
    accumulated_analysis["metadata"]["total_gold_reports"] = num_reports

    # Persist analysis
    output_dir = exp / "results" / "gold"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gold_analysis.json"

    logger.info(f"Saving analysis to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(accumulated_analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"Gold analysis saved to {output_path}")
    print(f"Gold analysis saved to {output_path}")
    print(json.dumps(accumulated_analysis, indent=2)[:3000])

    # Clean up checkpoint file on successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed after successful completion")


if __name__ == "__main__":
    main()