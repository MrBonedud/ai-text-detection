"""
Spot-check 50 samples each (AI and Human) to verify data quality
Compares before/after preprocessing and validates output cleanliness
"""

import pandas as pd
import os
import random
from datetime import datetime

def load_data():
    """Load raw and clean datasets"""
    raw_path = os.path.join('data', 'processed', 'raw_data.csv')
    clean_path = os.path.join('data', 'processed', 'clean_data.csv')
    
    print("Loading datasets...")
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(clean_path)
    
    assert len(raw_df) == len(clean_df), "Row count mismatch between raw and clean data!"
    print(f"✓ Loaded {len(raw_df)} samples from each dataset\n")
    
    return raw_df, clean_df

def validate_data_integrity(raw_df, clean_df):
    """Check for data corruption or alignment issues"""
    issues = []
    
    # Check row counts
    if len(raw_df) != len(clean_df):
        issues.append(f"Row count mismatch: raw={len(raw_df)}, clean={len(clean_df)}")
    
    # Check labels match
    if not (raw_df['label'].values == clean_df['label'].values).all():
        issues.append("Labels don't match between raw and clean")
    
    # Check for missing values
    raw_missing = raw_df['text'].isna().sum()
    clean_missing = clean_df['text'].isna().sum()
    if raw_missing > 0 or clean_missing > 0:
        issues.append(f"Missing values - raw: {raw_missing}, clean: {clean_missing}")
    
    # Check for empty strings after cleaning
    empty_clean = (clean_df['text'].str.len() == 0).sum()
    if empty_clean > 0:
        issues.append(f"Empty strings after cleaning: {empty_clean}")
    
    if issues:
        print("⚠ DATA INTEGRITY ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Data integrity check passed\n")
        return True

def spot_check_samples(raw_df, clean_df, num_samples=50):
    """Spot-check specified number of samples from each class"""
    
    results = {
        'ai': [],
        'human': []
    }
    
    # Process AI samples (label=1)
    ai_indices = raw_df[raw_df['label'] == 1].index.tolist()
    ai_samples = random.sample(ai_indices, min(num_samples, len(ai_indices)))
    
    # Process Human samples (label=0)
    human_indices = raw_df[raw_df['label'] == 0].index.tolist()
    human_samples = random.sample(human_indices, min(num_samples, len(human_indices)))
    
    print(f"Spot-checking {len(ai_samples)} AI samples and {len(human_samples)} Human samples...\n")
    
    # Check AI samples
    print("=" * 80)
    print(f"AI-GENERATED SAMPLES ({len(ai_samples)} spot-checks)")
    print("=" * 80)
    
    for i, idx in enumerate(ai_samples[:5]):  # Show first 5 in detail
        raw_text = raw_df.iloc[idx]['text']
        clean_text = clean_df.iloc[idx]['text']
        
        print(f"\n[Sample {i+1}]")
        print(f"BEFORE (Raw - {len(raw_text)} chars):")
        print(f"  {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")
        print(f"AFTER (Clean - {len(clean_text)} chars):")
        print(f"  {clean_text[:120]}{'...' if len(clean_text) > 120 else ''}")
        
        # Verify cleanliness
        checks = {
            'no_urls': 'http' not in clean_text and 'www' not in clean_text,
            'no_emails': '@' not in clean_text,
            'no_numbers': not any(c.isdigit() for c in clean_text),
            'no_uppercase': clean_text.islower() or clean_text == '',
            'has_content': len(clean_text) > 0,
        }
        
        status = "✓" if all(checks.values()) else "⚠"
        print(f"{status} Quality checks: {', '.join(k for k,v in checks.items() if v)}")
        
        results['ai'].append({
            'index': idx,
            'raw_len': len(raw_text),
            'clean_len': len(clean_text),
            'checks': checks,
            'passed': all(checks.values())
        })
    
    # Check Human samples
    print("\n" + "=" * 80)
    print(f"HUMAN SAMPLES ({len(human_samples)} spot-checks)")
    print("=" * 80)
    
    for i, idx in enumerate(human_samples[:5]):  # Show first 5 in detail
        raw_text = raw_df.iloc[idx]['text']
        clean_text = clean_df.iloc[idx]['text']
        
        print(f"\n[Sample {i+1}]")
        print(f"BEFORE (Raw - {len(raw_text)} chars):")
        print(f"  {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")
        print(f"AFTER (Clean - {len(clean_text)} chars):")
        print(f"  {clean_text[:120]}{'...' if len(clean_text) > 120 else ''}")
        
        # Verify cleanliness
        checks = {
            'no_urls': 'http' not in clean_text and 'www' not in clean_text,
            'no_emails': '@' not in clean_text,
            'no_numbers': not any(c.isdigit() for c in clean_text),
            'no_uppercase': clean_text.islower() or clean_text == '',
            'has_content': len(clean_text) > 0,
        }
        
        status = "✓" if all(checks.values()) else "⚠"
        print(f"{status} Quality checks: {', '.join(k for k,v in checks.items() if v)}")
        
        results['human'].append({
            'index': idx,
            'raw_len': len(raw_text),
            'clean_len': len(clean_text),
            'checks': checks,
            'passed': all(checks.values())
        })
    
    # Check all remaining samples (silent check)
    for idx in ai_samples[5:]:
        raw_text = raw_df.iloc[idx]['text']
        clean_text = clean_df.iloc[idx]['text']
        checks = {
            'no_urls': 'http' not in clean_text and 'www' not in clean_text,
            'no_emails': '@' not in clean_text,
            'no_numbers': not any(c.isdigit() for c in clean_text),
            'no_uppercase': clean_text.islower() or clean_text == '',
            'has_content': len(clean_text) > 0,
        }
        results['ai'].append({
            'index': idx,
            'raw_len': len(raw_text),
            'clean_len': len(clean_text),
            'checks': checks,
            'passed': all(checks.values())
        })
    
    for idx in human_samples[5:]:
        raw_text = raw_df.iloc[idx]['text']
        clean_text = clean_df.iloc[idx]['text']
        checks = {
            'no_urls': 'http' not in clean_text and 'www' not in clean_text,
            'no_emails': '@' not in clean_text,
            'no_numbers': not any(c.isdigit() for c in clean_text),
            'no_uppercase': clean_text.islower() or clean_text == '',
            'has_content': len(clean_text) > 0,
        }
        results['human'].append({
            'index': idx,
            'raw_len': len(raw_text),
            'clean_len': len(clean_text),
            'checks': checks,
            'passed': all(checks.values())
        })
    
    return results

def print_summary(results):
    """Print summary statistics"""
    ai_results = results['ai']
    human_results = results['human']
    
    ai_passed = sum(1 for r in ai_results if r['passed'])
    human_passed = sum(1 for r in human_results if r['passed'])
    
    ai_avg_raw = sum(r['raw_len'] for r in ai_results) / len(ai_results)
    ai_avg_clean = sum(r['clean_len'] for r in ai_results) / len(ai_results)
    human_avg_raw = sum(r['raw_len'] for r in human_results) / len(human_results)
    human_avg_clean = sum(r['clean_len'] for r in human_results) / len(human_results)
    
    print("\n" + "=" * 80)
    print("SPOT-CHECK SUMMARY")
    print("=" * 80)
    
    print(f"\nAI SAMPLES (n={len(ai_results)}):")
    print(f"  ✓ Passed quality checks: {ai_passed}/{len(ai_results)} ({100*ai_passed/len(ai_results):.1f}%)")
    print(f"  Average length before: {ai_avg_raw:.0f} chars")
    print(f"  Average length after: {ai_avg_clean:.0f} chars")
    print(f"  Reduction: {100*(ai_avg_raw-ai_avg_clean)/ai_avg_raw:.1f}%")
    
    print(f"\nHUMAN SAMPLES (n={len(human_results)}):")
    print(f"  ✓ Passed quality checks: {human_passed}/{len(human_results)} ({100*human_passed/len(human_results):.1f}%)")
    print(f"  Average length before: {human_avg_raw:.0f} chars")
    print(f"  Average length after: {human_avg_clean:.0f} chars")
    print(f"  Reduction: {100*(human_avg_raw-human_avg_clean)/human_avg_raw:.1f}%")
    
    overall_passed = ai_passed + human_passed
    overall_total = len(ai_results) + len(human_results)
    print(f"\nOVERALL:")
    print(f"  ✓ Total passed: {overall_passed}/{overall_total} ({100*overall_passed/overall_total:.1f}%)")
    
    if overall_passed == overall_total:
        print("\n✓✓✓ ALL SPOT-CHECKS PASSED - DATA IS CLEAN ✓✓✓")
    else:
        print(f"\n⚠ {overall_total - overall_passed} samples failed quality checks - review needed")
    
    print("=" * 80)

def main():
    print("\n" + "=" * 80)
    print("SPOT-CHECK: AI vs Human Text Dataset")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Load data
    raw_df, clean_df = load_data()
    
    # Validate integrity
    integrity_ok = validate_data_integrity(raw_df, clean_df)
    
    # Spot-check samples
    results = spot_check_samples(raw_df, clean_df, num_samples=50)
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Spot-check complete!\n")

if __name__ == '__main__':
    main()
