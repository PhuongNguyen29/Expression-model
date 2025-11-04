"""
Test script to verify StratifiedBatchSampler works correctly.
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from src.utils.batch_samplers import StratifiedBatchSampler

# Load your TCGA data
surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)

events = surv_tcga['event'].values
n_samples = len(events)
n_events = events.sum()

print(f"Dataset: {n_samples} samples, {n_events} events ({100*n_events/n_samples:.1f}%)")

# Test with your actual batch size
batch_size = 48

sampler = StratifiedBatchSampler(
    events=events,
    batch_size=batch_size,
    min_events_per_batch=1,
    shuffle=False  # False for testing to see deterministic output
)

print(f"\nTesting batch composition:")
print(f"{'Batch':<8} {'Size':<8} {'Events':<10} {'Event %':<10}")
print("-" * 40)

zero_event_batches = 0
for i, batch_indices in enumerate(sampler):
    batch_events = events[batch_indices].sum()
    event_pct = 100 * batch_events / len(batch_indices)
    
    print(f"{i+1:<8} {len(batch_indices):<8} {batch_events:<10} {event_pct:<10.1f}%")
    
    if batch_events == 0:
        zero_event_batches += 1

print("-" * 40)
print(f"Total batches: {i+1}")
print(f"Batches with zero events: {zero_event_batches}")

if zero_event_batches == 0:
    print("\n✅ SUCCESS: All batches contain at least one event!")
else:
    print(f"\n❌ FAILURE: {zero_event_batches} batches have no events")