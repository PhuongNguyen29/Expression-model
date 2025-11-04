# Biomarker Extraction - Execution Guide

## 📋 Pre-Flight Checklist

### Step 1: Verify Your Pipeline (5 minutes)

```bash
# Navigate to your project directory
cd ~/hpchome/Expression-model/

# Activate environment
conda activate Expression-model-env-py38

# Run verification script
python verify_pipeline.py
```

**Expected output**: "✅ ALL CHECKS PASSED!"

If any checks fail, see "Troubleshooting" section below.

---

### Step 2: Check Your Feature Selection Module

Your existing `src/utils/feature_selection.py` should have these functions:
- `compute_gene_importance_l2()`
- `select_features_percentile()`
- `get_selected_gene_names()`
- `compute_bidirectional_consensus()`
- `compare_with_chapter2_biomarkers()`

**To verify**:
```bash
python -c "from src.utils.feature_selection import compute_gene_importance_l2; print('✓ OK')"
```

**If you get ImportError**: Replace your `src/utils/feature_selection.py` with the complete version I provided (`feature_selection_complete.py`)

```bash
# Backup existing file
cp src/utils/feature_selection.py src/utils/feature_selection.py.backup

# Copy complete version (after downloading from this chat)
cp feature_selection_complete.py src/utils/feature_selection.py
```

---

## 🚀 Main Execution

### Step 3: Run Biomarker Extraction

**Basic command** (recommended):
```bash
python scripts/extract_biomarkers_from_best_params.py \
    --tcga_params results/20251104/hyperparam_tcga_20251104_034717/best_params.json \
    --orien_params results/20251104/hyperparam_orien_20251104_032058/best_params.json \
    --output_dir results/biomarker_extraction/ \
    --selection_method percentile \
    --percentile 95.0 \
    --n_epochs 150
```

**Expected runtime**: 30-45 minutes (mostly model training)

**What it does**:
1. Retrains TCGA model on 100% data (15-20 min)
2. Retrains ORIEN model on 100% data (15-20 min)
3. Extracts feature importance (< 1 min)
4. Finds consensus genes (< 1 min)
5. Compares with Chapter 2 genes (< 1 min)

---

### Step 4: Monitor Progress

The script will print progress like:
```
============================================================
TRAINING TCGA MODEL
============================================================
Samples: 339
Raw genes: 20516
Genes after preprocessing: 308
Architecture: 308 → 256 → 64 → 1
Total parameters: 82,816
Training on: cuda
Training for 150 epochs...
Epoch 10/150: Train Loss: 5.234, Train C-index: 0.621
...
```

---

## 📊 Expected Results

### Output Files

```
results/biomarker_extraction_YYYYMMDD_HHMMSS/
├── all_gene_importances.csv          # All 308 genes ranked
├── tcga_selected_genes.csv           # Top 5% from TCGA (~15 genes)
├── orien_selected_genes.csv          # Top 5% from ORIEN (~15 genes)
├── consensus_genes.csv               # Intersection (⭐ KEY FILE)
├── chapter2_comparison.json          # Overlap with Cox genes
├── tcga_model.pth                    # Trained models
├── orien_model.pth
└── SUMMARY.json                      # Complete results
```

### Expected Consensus Results

**Scenario 1: Good Stability** (15-30% overlap)
```json
{
  "n_consensus": 5-10,
  "jaccard_index": 0.25-0.40,
  "overlap_rate": 0.33-0.50
}
```

**Scenario 2: Moderate Stability** (10-15% overlap)
```json
{
  "n_consensus": 2-5,
  "jaccard_index": 0.10-0.25
}
```

**Scenario 3: Poor Stability** (<10% overlap)
```json
{
  "n_consensus": 0-2,
  "jaccard_index": 0.0-0.10
}
```

**Important**: Even poor stability is a valid scientific finding! It indicates neural networks don't identify stable biomarkers in small-sample settings.

---

## 🔧 Troubleshooting

### Error: "No module named 'src.utils.feature_selection'"

**Solution**: Make sure you're in the project root directory
```bash
cd ~/hpchome/Expression-model/
python scripts/extract_biomarkers_from_best_params.py ...
```

### Error: "Cannot import 'compute_gene_importance_l2'"

**Solution**: Your `feature_selection.py` is missing functions. Replace it:
```bash
cp feature_selection_complete.py src/utils/feature_selection.py
```

### Error: "consensus_genes_308.txt not found"

**Solution**: Check the actual filename
```bash
ls data/raw/consensus_genes*
```

Then update the command:
```bash
--chapter2_genes data/raw/YOUR_ACTUAL_FILENAME.txt
```

### Error: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU
```bash
# Edit best_params.json to reduce batch_size
# Or the script will automatically fall back to CPU
```

### Warning: "No consensus genes found"

**This is not an error!** It means:
- Neural networks identified completely different genes in each cohort
- Valid scientific finding about feature instability
- Still proceed to next steps

---

## 📝 After Completion

### Check Your Results

```bash
# View consensus genes
cat results/biomarker_extraction_*/consensus_genes.csv

# View summary
cat results/biomarker_extraction_*/SUMMARY.json

# Check overlap with Chapter 2
cat results/biomarker_extraction_*/chapter2_comparison.json
```

### Interpret Results

**If you got 5-15 consensus genes**:
✅ Good! Proceed to bidirectional validation using these genes

**If you got 0-4 consensus genes**:
✅ Also valid! This is a scientific finding about neural network instability
⚠️ Consider using all 308 genes for bidirectional validation instead

---

## 🎯 Next Steps

### Option A: Bidirectional Validation with Consensus Genes
```bash
python evaluate_bidirectional_FINAL.py \
    --tcga_params results/20251104/hyperparam_tcga_*/best_params.json \
    --orien_params results/20251104/hyperparam_orien_*/best_params.json \
    --consensus_genes results/biomarker_extraction_*/consensus_genes.csv \
    --output_dir results/bidirectional_consensus/
```

### Option B: Bidirectional Validation with All 308 Genes
```bash
python evaluate_bidirectional_FINAL.py \
    --tcga_params results/20251104/hyperparam_tcga_*/best_params.json \
    --orien_params results/20251104/hyperparam_orien_*/best_params.json \
    --output_dir results/bidirectional_308genes/
```

### Option C: Try Different Selection Threshold
```bash
# Use top 10% instead of top 5%
python scripts/extract_biomarkers_from_best_params.py \
    --percentile 90.0 \
    ...
```

---

## 📊 For Your Advisor Meeting

Present these results:

1. **Hyperparameter tuning complete** ✅
   - TCGA CV C-index: 0.671
   - ORIEN CV C-index: 0.641

2. **Biomarker extraction complete** ✅
   - TCGA selected: X genes
   - ORIEN selected: Y genes
   - Consensus: Z genes
   - Overlap with Chapter 2: W genes (W/20 = %)

3. **Next steps**:
   - Bidirectional validation with consensus genes
   - Compare final C-index with Chapter 2 (0.68 / 0.72)

---

## 💡 Quick Commands Reference

```bash
# Full workflow
cd ~/hpchome/Expression-model/
conda activate Expression-model-env-py38

# Verify
python verify_pipeline.py

# Extract biomarkers
python scripts/extract_biomarkers_from_best_params.py \
    --tcga_params results/20251104/hyperparam_tcga_20251104_034717/best_params.json \
    --orien_params results/20251104/hyperparam_orien_20251104_032058/best_params.json \
    --output_dir results/biomarker_extraction/

# Check results
cat results/biomarker_extraction_*/SUMMARY.json
```

---

## 📞 Need Help?

If you encounter issues:
1. Share the exact error message
2. Share output of `verify_pipeline.py`
3. Share your SUMMARY.json content

Good luck! 🚀
