#!/usr/bin/env python3
"""
Quick verification: Check if model files exist for k-sweep
"""

from pathlib import Path

# Your models directory (where the .pth files are stored)
MODELS_DIR = "results/transfer_learning"
SEEDS = [42, 123, 456, 789, 1011]

print("="*60)
print("QUICK VERIFICATION: Model Files Check")
print("="*60)
print(f"\nModels directory: {MODELS_DIR}\n")

models_dir = Path(MODELS_DIR)

if not models_dir.exists():
    print(f"❌ ERROR: Directory not found!")
    print(f"   {models_dir.absolute()}")
    exit(1)

print("✓ Models directory exists\n")

# Check each seed
tcga_found = 0
orien_found = 0

for seed in SEEDS:
    print(f"Seed {seed}:")
    
    # Check ORIEN→TCGA
    orien_to_tcga_pattern = f"orien_to_tcga_seed{seed}_*"
    orien_dirs = list(models_dir.glob(orien_to_tcga_pattern))
    
    if orien_dirs:
        model_path = orien_dirs[0] / f"tcga_finetuned_seed{seed}.pth"
        if model_path.exists():
            print(f"  ✓ ORIEN→TCGA: {model_path.name}")
            tcga_found += 1
        else:
            print(f"  ✗ ORIEN→TCGA: Directory found but .pth missing")
    else:
        print(f"  ✗ ORIEN→TCGA: Not found")
    
    # Check TCGA→ORIEN
    tcga_to_orien_pattern = f"tcga_to_orien_seed{seed}_*"
    tcga_dirs = list(models_dir.glob(tcga_to_orien_pattern))
    
    if tcga_dirs:
        model_path = tcga_dirs[0] / f"orien_finetuned_seed{seed}.pth"
        if model_path.exists():
            print(f"  ✓ TCGA→ORIEN: {model_path.name}")
            orien_found += 1
        else:
            print(f"  ✗ TCGA→ORIEN: Directory found but .pth missing")
    else:
        print(f"  ✗ TCGA→ORIEN: Not found")
    
    print()

print("="*60)
print("SUMMARY")
print("="*60)
print(f"ORIEN→TCGA models: {tcga_found}/{len(SEEDS)}")
print(f"TCGA→ORIEN models: {orien_found}/{len(SEEDS)}")
print(f"Total models: {tcga_found + orien_found}/{len(SEEDS)*2}")
print()

if tcga_found == len(SEEDS) and orien_found == len(SEEDS):
    print("✅ ALL MODEL FILES FOUND!")
    print("\nYou're ready to run the k-sweep:")
    print("  chmod +x run_biomarker_ksweep.sh")
    print("  ./run_biomarker_ksweep.sh")
elif tcga_found > 0 or orien_found > 0:
    print("⚠️  PARTIAL: Some model files are missing")
    print("\nK-sweep will run but may have incomplete results.")
    print("Consider investigating missing models.")
else:
    print("❌ ERROR: No model files found!")
    print("\nPlease check:")
    print(f"  1. Directory path: {models_dir.absolute()}")
    print("  2. Model file naming pattern")
