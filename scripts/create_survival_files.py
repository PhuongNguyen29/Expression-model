import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_survival_files():
    """Create the missing harmonized survival files"""
    
    logger.info("Creating harmonized survival files...")
    
    # Run data loader to get harmonized data
    loader = DataLoader()
    data = loader.run_full_pipeline()
    
    # Save survival data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save survival data with sample IDs as index
    logger.info(f"Saving survival data to {processed_dir}")
    data['surv_tcga'].to_csv(processed_dir / "surv_tcga_harmonized.csv")
    data['surv_orien'].to_csv(processed_dir / "surv_orien_harmonized.csv")
    
    # Verify files were created
    files_created = [
        processed_dir / "surv_tcga_harmonized.csv",
        processed_dir / "surv_orien_harmonized.csv"
    ]
    
    for file in files_created:
        if file.exists():
            df = pd.read_csv(file, index_col=0)
            logger.info(f"✓ Created {file.name}: {len(df)} samples")
        else:
            logger.error(f"✗ Failed to create {file.name}")
    
    logger.info("Done!")
    return data

if __name__ == "__main__":
    create_survival_files()
    print("\nNow check: ls data/processed/")
    print("You should see surv_tcga_harmonized.csv and surv_orien_harmonized.csv")