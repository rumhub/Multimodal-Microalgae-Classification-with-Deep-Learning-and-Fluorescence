import pandas as pd
from pathlib import Path
from . import config

class CSVWriter:
    def __init__(self):
        pass
    
    # @brief: Export images information into a csv
    def export_img_info_to_csv(self, img_paths, output_path):
        print("--------- CSV WRITER ----------------")
        print(f"Exporting data as {Path(output_path).resolve()}\n")

        rows = []
        
        # For every image
        for base_name, fields in img_paths.items():
            
            # Add image name as base name
            row = {"base_name": base_name}
            
            # Add every field except channel names
            for k, v in fields.items():
                if k not in config.IMG_SUFFIXES:
                    row[k] = v
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)