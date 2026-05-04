import pandas as pd
from pathlib import Path
from . import config

class CSVWriter:
    def __init__(self):
        pass
    
    # @brief: Export images information into a csv
    def export_img_info_to_csv(self, img_paths, output_path):
        print(f"Exporting data as {Path(output_path).resolve()}\n")
    
        rows = []
    
        for base_name, fields in img_paths.items():
            row = {"base_name": base_name}
    
            for k, v in fields.items():
                if k not in config.IMG_SUFFIXES:
                    row[k] = v
    
            rows.append(row)
    
        df = pd.DataFrame(rows)
    
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        # Export image-level data
        df.to_csv(output_path, index=False)
    
        # Numeric columns, excluding labels / identifiers
        excluded_summary_cols = ["class", "base_name"]
        numeric_cols = [
            col for col in df.select_dtypes(include="number").columns
            if col not in excluded_summary_cols
        ]
    
        if len(numeric_cols) == 0:
            return
    
        with open(output_path, "a", newline="") as f:
            # =================== SUMMARY BY CLASS ===================
            if "class" in df.columns:
                f.write("\n")
                f.write("SUMMARY BY CLASS\n")
    
                for class_label in sorted(df["class"].dropna().unique()):
                    class_df = df[df["class"] == class_label]
                    numeric_df = class_df[numeric_cols]
    
                    if numeric_df.empty:
                        continue
    
                    summary_df = numeric_df.describe().T
    
                    if "count" in summary_df.columns:
                        summary_df = summary_df.drop(columns=["count"])
    
                    f.write(f"\nCLASS {class_label}\n")
                    f.write(f"num_algae,{len(class_df)}\n")
                    summary_df.to_csv(f, index=True, index_label="metric")
    
            # =================== GLOBAL SUMMARY ===================
            f.write("\n")
            f.write("GLOBAL SUMMARY\n")
            f.write(f"num_algae,{len(df)}\n")
    
            global_summary_df = df[numeric_cols].describe().T
    
            if "count" in global_summary_df.columns:
                global_summary_df = global_summary_df.drop(columns=["count"])
    
            global_summary_df.to_csv(f, index=True, index_label="metric")
        
        
    def export_summary_to_csv(self, summary_df, output_path):
        print(f"Exporting summary as {Path(output_path).resolve()}\n")
    
        output_path = Path(output_path)
    
        # Create parent folder if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        summary_df.to_csv(output_path, index=True)