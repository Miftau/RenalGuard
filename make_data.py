import pandas as pd
import os

# Set working directory to where files are located
data_dir = "nchs_data"
os.chdir(data_dir)

# List ALL available XPT files from your directory
xpt_files = [
    'WHQ_L.xpt', 'SMQRTU_L.xpt', 'SMQFAM_L.xpt', 'SMQ_L.xpt', 'SLQ_L.xpt',
    'RHQ_L.xpt', 'RXQASA_L.xpt', 'RXQ_RX_L.xpt', 'PAQY_L.xpt', 'PAQ_L.xpt',
    'PUQMEC_L.xpt', 'OHQ_L.xpt', 'OCQ_L.xpt', 'DPQ_L.xpt', 'MCQ_L.xpt',
    'KIQ_U_L.xpt', 'INQ_L.xpt', 'IMQ_L.xpt', 'HOQ_L.xpt', 'HUQ_L.xpt',
    'FNQ_L.xpt', 'ECQ_L.xpt', 'DBQ_L.xpt', 'DIQ_L.xpt', 'DEQ_L.xpt',
    'HSQ_L.xpt', 'BPQ_L.xpt', 'BAQ_L.xpt', 'ALQ_L.xpt', 'ACQ_L.xpt',
    'BMX_L.xpt', 'BPXO_L.xpt', 'DEMO_L.xpt', 'GLU_L.xpt', 'GHB_L.xpt',
    'HDL_L.xpt', 'TCHOL_L.xpt', 'TRIGLY_L.xpt',
    'BAX_L.xpt', 'LUX_L.xpt', 'BMX_L.xpt', 'BPXO_L.xpt', 'DEMO_L.xpt',
    'BIOPRO_L.xpt', 'TST_L.xpt', 'FOLFMS_L.xpt', 'GLU_L.xpt', 'IHGEM_L.xpt',
    'PBCD_L.xpt', 'INS_L.xpt', 'HSCRP_L.xpt', 'HEPB_S_L.xpt', 'HEPA_L.xpt',
    'GHB_L.xpt', 'FOLATE_L.xpt', 'FERTIN_L.xpt', 'FASTQX_L.xpt', 'TCHOL_L.xpt',
    'CBC_L.xpt', 'TRIGLY_L.xpt', 'HDL_L.xpt', 'AGP_L.xpt',
    'HOQ_L.xpt', 'HUQ_L.xpt', 'HEQ_L.xpt', 'HIQ_L.xpt', 'FNQ_L.xpt',
    'ECQ_L.xpt', 'DBQ_L.xpt', 'DIQ_L.xpt', 'DEQ_L.xpt', 'HSQ_L.xpt',
    'BPQ_L.xpt', 'BAQ_L.xpt', 'ALQ_L.xpt', 'ACQ_L.xpt', 'AUQ_L.xpt',
    'VOCWB_L.xpt', 'VID_L.xpt', 'UCPREG_L.xpt', 'TFR_L.xpt'
]

# Remove duplicates (some appear twice)
xpt_files = list(set(xpt_files))

# Verify existence
missing = [f for f in xpt_files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing files: {missing}")

print("All files found. Starting merge...")

# Initialize the merged DataFrame with the first file
try:
    base_df = pd.read_sas(xpt_files[0], format='xport')
    base_df['SEQN'] = pd.to_numeric(base_df['SEQN'], errors='coerce').astype('Int64')
    print(f"Loaded {xpt_files[0]}. Shape: {base_df.shape}")

    # Merge the remaining files iteratively
    for file in xpt_files[1:]:
        print(f"Merging {file}...")
        try:
            current_df = pd.read_sas(file, format='xport')
            current_df['SEQN'] = pd.to_numeric(current_df['SEQN'], errors='coerce').astype('Int64')

            # Use suffixes to handle potential duplicates
            # Only merge on 'SEQN', suffixes will be added to other overlapping columns automatically
            base_df = pd.merge(base_df, current_df, on='SEQN', how='outer', suffixes=('', '_y'))
            
            # Identify and handle duplicate columns after merge
            # Columns like WTPH2YR_x, WTPH2YR_y may exist if WTPH2YR was in both base and current.
            # We will keep the one from the current (right) dataset, assuming it's more specific or newer.
            # This is a common approach for overlapping weights or flags.
            # Find columns ending with '_y'
            y_cols = [col for col in base_df.columns if col.endswith('_y')]
            x_cols = [col.replace('_y', '') for col in y_cols] # corresponding '_x' names
            
            for x_col, y_col in zip(x_cols, y_cols):
                if x_col in base_df.columns:
                    # Fill NaN values in the original column with values from the suffixed one
                    base_df[x_col] = base_df[x_col].fillna(base_df[y_col])
                    # Drop the suffixed column
                    base_df.drop(columns=[y_col], inplace=True)
                    print(f"  - Merged {y_col} into {x_col}, dropped {y_col}")

        except Exception as e:
            print(f"Error reading or merging {file}: {e}")
            print("Skipping this file.")

    print(f"\nFinal merged dataset shape: {base_df.shape}")

    # --- Data Cleaning ---
    print("\nStarting data cleaning...")

    # Define missing value codes (NHANES standard)
    missing_codes = [-9, -8, -7, -1]

    # Apply to all numeric columns
    numeric_cols = base_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        base_df[col] = base_df[col].replace(missing_codes, pd.NA)

    # Drop rows with invalid age or BMI
    base_df = base_df[(base_df['RIDAGEYR'] >= 1) & (base_df['RIDAGEYR'] <= 120)]
    if 'BMXBMI' in base_df.columns:
        base_df = base_df[(base_df['BMXBMI'] >= 10) & (base_df['BMXBMI'] <= 100)]

    # Recode categorical variables (optional: map to readable labels)
    # Example: Gender
    if 'RIAGENDR' in base_df.columns:
        gender_map = {1: 'Male', 2: 'Female'}
        base_df['RIAGENDR'] = base_df['RIAGENDR'].map(gender_map).astype('category')

    print("Data cleaning complete.")

    # --- Define Feature Sets (Only variables that exist in your files) ---
    print("\nSelecting features...")

    # === KIDNEY ===
    kidney_general_vars = [
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL',
        'BMXBMI', 'BMXWAIST',
        'SMQ020', 'SMQ040', 'SMQ077',  # smoking
        'ALQ101', 'ALQ130',  # alcohol
        'PADACTTV', 'PAQ605',  # physical activity
        'DIQ010',  # diabetes self-report
        'BPQ020',  # hypertension self-report
        'MCQ160E'  # family history of kidney disease
    ]

    kidney_clinical_vars = [
        'SEQN',
        'LBXSCR', 'LBXBUN', 'LBXEGFR',  # kidney function markers
        'LBXGLU', 'LBXGH',  # glucose/HbA1c
        'LBDHDD', 'LBDLDL', 'LBXTCA',  # lipids
        'BPXSY1', 'BPXDI1',  # systolic/diastolic BP
        'LBXPLTSI'  # platelets (from CBC_L.xpt)
    ]

    # === LIVER ===
    liver_general_vars = [
        'SEQN',
        'ALQ101', 'ALQ130', 'ALQ151',  # alcohol frequency/quantity
        'BMXBMI', 'BMXWAIST',
        'DIQ010',  # diabetes
        'RXQASAS1',  # NSAID/aspirin use
        'SMQ020',  # smoking
        'DBD100', 'DBD895',  # diet
        'PADACTTV', 'PAQ605',  # activity
        'RIDAGEYR', 'RIAGENDR'
    ]

    liver_clinical_vars = [
        'SEQN',
        'LBXALT', 'LBXAST', 'LBXGGT',  # liver enzymes (from HEPB_S_L.xpt or HEPA_L.xpt)
        'LBXTB',  # total bilirubin
        'LBXALBSI',  # albumin
        'LBXPLTSI',  # platelets
        'LBXTR', 'LBXGLU', 'LBXGH',  # triglycerides, glucose, HbA1c
        'LBDHDD', 'LBXTCA'  # HDL, total cholesterol
    ]

    # Helper: safe column selection
    def safe_select(df, cols):
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    # Create datasets
    kidney_gen_df = safe_select(base_df, kidney_general_vars)
    kidney_cli_df = safe_select(base_df, kidney_clinical_vars)
    liver_gen_df = safe_select(base_df, liver_general_vars)
    liver_cli_df = safe_select(base_df, liver_clinical_vars)

    # --- Create Binary Outcome Variables ---
    print("\nCreating outcome variables...")

    # Kidney Disease Outcome: Ever told by doctor you have kidney disease
    if 'KIQ022' in base_df.columns:
        kidney_gen_df['KIDNEY_DISEASE'] = (base_df['KIQ022'] == 1).astype(int)
        kidney_cli_df['KIDNEY_DISEASE'] = (base_df['KIQ022'] == 1).astype(int)
        print("  - KIDNEY_DISEASE outcome added (KIQ022).")
    else:
        print("  - WARNING: KIQ022 not found, no KIDNEY_DISEASE outcome created.")

    # Liver Injury Proxy: ALT > 40 U/L (common clinical cutoff)
    if 'LBXALT' in base_df.columns:
        liver_gen_df['LIVER_INJURY'] = (base_df['LBXALT'] > 40).astype(int)
        liver_cli_df['LIVER_INJURY'] = (base_df['LBXALT'] > 40).astype(int)
        print("  - LIVER_INJURY outcome added (LBXALT > 40).")
    else:
        print("  - WARNING: LBXALT not found, no LIVER_INJURY outcome created.")

    # --- Save to CSV (outside nchs_data folder) ---
    output_dir = ".."  # parent directory
    kidney_gen_df.to_csv(os.path.join(output_dir, 'nhanes_kidney_general.csv'), index=False)
    kidney_cli_df.to_csv(os.path.join(output_dir, 'nhanes_kidney_clinical.csv'), index=False)
    liver_gen_df.to_csv(os.path.join(output_dir, 'nhanes_liver_general.csv'), index=False)
    liver_cli_df.to_csv(os.path.join(output_dir, 'nhanes_liver_clinical.csv'), index=False)

    print("\n✅ Successfully saved 4 cleaned datasets:")
    print("1. nhanes_kidney_general.csv")
    print("2. nhanes_kidney_clinical.csv")
    print("3. nhanes_liver_general.csv")
    print("4. nhanes_liver_clinical.csv")

    # Print sample info
    print("\n--- Sample Info ---")
    print("Kidney General Shape:", kidney_gen_df.shape)
    print("Kidney Clinical Shape:", kidney_cli_df.shape)
    print("Liver General Shape:", liver_gen_df.shape)
    print("Liver Clinical Shape:", liver_cli_df.shape)

except FileNotFoundError:
    print("Error: Ensure all XPT files are in the 'nchs_data' directory relative to the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()