import pandas as pd
import numpy as np

# --- Configuration ---
# Replace these with the actual paths on your local machine
PATH_CSV_1 = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions/fe2_lgb_orig_80_grin_all_19.csv'
PATH_CSV_2 = '/Users/malaymishra/Desktop/quant_ml_project/blend_lgb80_grin20.csv' 

def compare_submissions(file1, file2, tolerance=1e-8):
    print(f"Loading files...")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}\n")
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Ensure both have the expected columns
    for df in [df1, df2]:
        if not {'ID', 'TARGET'}.issubset(df.columns):
            raise ValueError("CSV files must contain 'ID' and 'TARGET' columns.")
            
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Merge on ID to guarantee we are comparing the exact same rows
    merged = pd.merge(df1, df2, on='ID', suffixes=('_local', '_kaggle'))
    
    if len(merged) != len(df1) or len(merged) != len(df2):
        print("\n[WARNING] Row counts do not match! The IDs differ between the files.")
        return
        
    # Extract the prediction arrays
    pred_local = merged['TARGET_local'].values
    pred_kaggle = merged['TARGET_kaggle'].values
    
    # 1. Exact Matches
    exact_matches = np.sum(pred_local == pred_kaggle)
    
    # 2. Tolerance Matches (Floating point math on Mac vs Linux differs slightly)
    abs_diff = np.abs(pred_local - pred_kaggle)
    close_matches = np.sum(abs_diff <= tolerance)
    
    # 3. Summary Statistics
    max_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # 4. Correlation (Should be 1.0)
    correlation = np.corrcoef(pred_local, pred_kaggle)[0, 1]
    
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"Total Rows Evaluated: {len(merged):,}")
    print(f"Exact Matches:        {exact_matches:,} ({(exact_matches/len(merged))*100:.2f}%)")
    print(f"Matches within {tolerance}: {close_matches:,} ({(close_matches/len(merged))*100:.2f}%)")
    print("-" * 40)
    print(f"Maximum Difference:   {max_diff:.8e}")
    print(f"Mean Abs Difference:  {mean_abs_diff:.8e}")
    print(f"Pearson Correlation:  {correlation:.6f}")
    print("="*40)
    
    if correlation > 0.999999:
        print("\n✅ The files are functionally identical.")
    elif correlation > 0.99:
        print("\n⚠️ The files are highly correlated but have some variance.")
    else:
        print("\n❌ The files are significantly different. Check the models.")

if __name__ == "__main__":
    compare_submissions(PATH_CSV_1, PATH_CSV_2)