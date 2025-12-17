import pandas as pd
import numpy as np

def run_validation_agent(file_a_path, file_b_path, output_path="validation_results.csv"):
    # 1. Load the CSVs
    df_a = pd.read_csv(file_a_path)
    df_b = pd.read_csv(file_b_path)

    # 2. Define the weighted importance of each field
    # Total sum = 1.0
    weights = {
        'NPI': 0.4,
        'Provider Name(Legal name)': 0.1,
        'Address': 0.05,
        'Phone number': 0.1,
        'License number': 0.35
    }

    # 3. Clean Data (Standardize for comparison)
    # We strip spaces and uppercase strings so "npi123" matches "NPI123 "
    for col in weights.keys():
        for df in [df_a, df_b]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper().replace(['NAN', 'NONE', ''], np.nan)

    # 4. Initialize scoring columns
    df_a['confidence_score'] = 1.0
    df_a['mismatch_reasons'] = ""

    # 5. Comparison Logic
    for field, weight in weights.items():
        if field not in df_a.columns or field not in df_b.columns:
            continue
            
        # Logic: 
        # A_exists = True if File A is NOT null
        # Mismatch = True if A != B
        a_exists = df_a[field].notnull()
        mismatch = df_a[field] != df_b[field]

        # Trigger deduction ONLY if value exists in A AND it doesn't match B
        deduct_mask = a_exists & mismatch

        # Apply the weight reduction
        df_a.loc[deduct_mask, 'confidence_score'] -= weight
        
        # Log the specific field failure
        df_a.loc[deduct_mask, 'mismatch_reasons'] += f"{field} mismatch; "

    # 6. Final Clean up
    df_a['confidence_score'] = df_a['confidence_score'].round(2).clip(lower=0)
    
    # Categorize results
    def categorize(score):
        if score == 1.0: return "Perfect Match"
        if score >= 0.7: return "Partial Match (Minor Issues)"
        return "Critical Mismatch"

    df_a['validation_status'] = df_a['confidence_score'].apply(categorize)

    # 7. Export results
    df_a.to_csv(output_path, index=False)
    print(f"Validation Agent: Processed {len(df_a)} rows. Report saved to {output_path}")
    
    return df_a

# Example Usage:
# results = run_validation_agent('provider_source.csv', 'provider_target.csv')