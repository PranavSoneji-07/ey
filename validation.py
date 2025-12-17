import pandas as pd
import numpy as np

def run_validation_agent(file_a_path, file_b_path, output_path="validation_results.csv"):
    # 1. Load the CSVs
    df_a = pd.read_csv(file_a_path)
    df_b = pd.read_csv(file_b_path)

    # 2. Define Weights with "Standardized" keys
    # We will normalize these keys to lowercase/no-space for matching
    raw_weights = {
        'npi_number': 0.4,
        'Provider Name(Legal name)': 0.1,
        'Address': 0.05,
        'phone': 0.1,
        'license_id': 0.35
    }
    
    # Helper to clean column names (e.g., "Phone Number" -> "phonenumber")
    def clean_col_name(name):
        return "".join(filter(str.isalnum, str(name))).lower()

    # Standardize weights dictionary keys
    weights = {clean_col_name(k): v for k, v in raw_weights.items()}
    
    # Standardize DataFrame column names for the duration of the script
    original_cols_a = df_a.columns.tolist()
    df_a.columns = [clean_col_name(c) for c in df_a.columns]
    df_b.columns = [clean_col_name(c) for c in df_b.columns]

    # 3. Drop columns from A that are completely empty
    df_a = df_a.dropna(axis=1, how='all')

    # 4. Initialize scoring
    df_a['confidence_score'] = 1.0
    df_a['mismatch_reasons'] = ""

    # 5. Comparison Logic
    checked = []
    for clean_field, weight in weights.items():
        if clean_field in df_a.columns and clean_field in df_b.columns:
            checked.append(clean_field)
            
            # Convert to string, strip, and uppercase for a "fair" comparison
            def normalize_val(val):
                if pd.isna(val) or str(val).strip().lower() in ['nan', 'none', '']:
                    return None
                return str(val).strip().upper()

            series_a = df_a[clean_field].apply(normalize_val)
            series_b = df_b[clean_field].apply(normalize_val)

            # Rule: IF A has data AND it doesn't match B
            mismatch = (series_a.notnull()) & (series_a != series_b)

            df_a.loc[mismatch, 'confidence_score'] -= weight
            df_a.loc[mismatch, 'mismatch_reasons'] += f"{clean_field} mismatch; "

    # 6. Final cleanup and restore original column names for the output
    df_a['confidence_score'] = df_a['confidence_score'].round(2).clip(lower=0)
    
    # Map back to original column names for the report
    column_mapping = {clean_col_name(original): original for original in original_cols_a}
    df_a.rename(columns=column_mapping, inplace=True)

    # 7. Export
    df_a.to_csv(output_path, index=False)
    print(f"--- Validation Complete ---")
    print(f"Processed: {len(df_a)} rows")
    print(f"Matched & Checked Columns: {checked}")
    
    if not checked:
        print("CRITICAL: No columns matched! Check if your CSV headers match the weights.")
    
    return df_a

# Run it
run_validation_agent('provider_output.csv', 'npi_batch_results.csv')