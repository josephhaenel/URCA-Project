import os
import pandas as pd

def save_df_to_excel(df, output_dir, file_name, sheet_name):
    os.makedirs(output_dir, exist_ok=True)  # This line will create the directory if it doesn't exist
    file_path = os.path.join(output_dir, file_name)
    mode = 'a' if os.path.exists(file_path) else 'w'
    if mode == 'a':
        with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)