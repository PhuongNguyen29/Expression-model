import sys
sys.path.append('.')  # Add project root to path

from src.data.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_data_loader():
    """Test the data loading pipeline"""
    print("="*60)
    print("TESTING DATA LOADER")
    print("="*60)
    
    # Initialize and run
    loader = DataLoader(config_path="config/default_config.yaml")
    data = loader.run_full_pipeline()
    
    # Verify outputs
    print("\n" + "="*60)
    print("VERIFICATION CHECKS")
    print("="*60)
    
    # Check shapes match
    assert data['tcga_expr'].shape[0] == data['orien_expr'].shape[0], \
        "Gene counts don't match between cohorts!"
    
    # Check sample counts match survival data
    assert data['tcga_expr'].shape[1] == len(data['surv_tcga']), \
        "TCGA sample count doesn't match survival data!"
    
    assert data['orien_expr'].shape[1] == len(data['surv_orien']), \
        "ORIEN sample count doesn't match survival data!"
    
    # Check survival data has required columns
    assert 'time' in data['surv_tcga'].columns, "Missing 'time' in TCGA survival"
    assert 'event' in data['surv_tcga'].columns, "Missing 'event' in TCGA survival"
    
    print("✓ All shape checks passed")
    print(f"✓ {len(data['common_genes'])} common genes identified")
    print(f"✓ TCGA: {data['tcga_expr'].shape[1]} samples")
    print(f"✓ ORIEN: {data['orien_expr'].shape[1]} samples")
    
    # Check value ranges
    print(f"\n✓ TCGA expression range: [{data['tcga_expr'].min().min():.2f}, {data['tcga_expr'].max().max():.2f}]")
    print(f"✓ ORIEN expression range: [{data['orien_expr'].min().min():.2f}, {data['orien_expr'].max().max():.2f}]")
    
    print("\n" + "="*60)
    print("DATA LOADER TEST PASSED!")
    print("="*60)
    
    return data

if __name__ == "__main__":
    data = test_data_loader()