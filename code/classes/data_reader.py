from pathlib import Path
from . import config

class DataReader:
    
    def __init__(self, path):
        self.path = path

    
    """
    @brief: Reads images and stores their paths + corresponding labels
    Dataset structure:
        base_dir:
            - Chlorella...
            - Chlorella...
            - Haematococus...
            - Scenedesmus...
            - Scenedesmus...
            - Haematococus...
        Folders are not ordered by name, so we'll map folder's initial name "Ch" to Chlorella imgs,
        "Ha" to Haematococcus imgs, and so on.
        
        Each microalgae is given by 7 different images.
        Images format for a single microalgae:
          - Common_id + Different_suffix
          - Possible suffixes: ["amp", "flr_1", "flr_2", "flr_3", "flu", "mask", "phase"]
    """
    def read_data(self):
        print("--------- IMAGE READING ----------------")

        # Define outpuut data structure
        img_paths = {}

        # Get base path to read from
        base_dir = Path(self.path)

        # Folders structure:
        # Inside the following folders there are 3 different folders, eacah of them containing images from Chlorella
        # This data comes from a classifier that doesn't work properly, our job here is precisely that, to create a proper one
        # Base_path:
        #  - Chlorella-20260126T182158Z-3-001/Chlorella/:
        #      - class_chlorella
        #      - class_haematococcus
        #      - class_smallparticle
        # - Haematococcus_verde1-20260126T202441Z-3-001/Haematococcus_verde1/:
        #      - ...
        
        print(f"Reading images from {base_dir.resolve()}\n")
        print("----")
        # Iterate over top level folders (Chlorella-20260126T182158Z-3-001, Haematococcus_verde1-20260126T202441Z-3-001, ...)
        for base_class_dir in base_dir.iterdir():
            
            # Get folder prefix
            class_dir_prefix = base_class_dir.name[:2].upper()
            
            # Go only into those folders that match our class prefixes
            if class_dir_prefix in config.CLASS_PREFIXES:
                
                # Get label corresponding to this class
                class_label = config.CLASS_PREFIXES[class_dir_prefix]
                print(f"Reading as {config.CLASS_NAMES[class_dir_prefix]} images from folder: {base_class_dir.name}")

                # Here we have to go into another folder
                # Example from above: Chlorella-20260126T182158Z-3-001/Chlorella/ --> Go into folder Chlorella
                subdir = next((d for d in base_class_dir.iterdir() if d.is_dir()), None)
                
                if subdir != None:
                    # print(f"Entering folder {subdir.name}")
                    
                    # Here we have to enter folders that start with "class"
                    # In the example above, these folders were class_chlorella, class_haematococcus, class_smallparticle, ...
                    for class_dir in subdir.iterdir():
                        if class_dir.name.startswith("class"):
                            
                            # Enter the folder and read all its images as class {class_label}
                            for img in class_dir.iterdir():
                                # print(f"Loading microalga {img.name}")
                                
                                # Remove .png
                                name = img.stem
                                # print(f"Image name {name}")
                                
                                ########################################
                                ## GROUP IMAGE CHANNELS PER MICROALGA ##
                                ########################################
                                
                                for suffix in config.IMG_SUFFIXES:
                                    if name.endswith(suffix):
                                        # Get base image name 
                                        # Example:
                                        #   - Alm_1_fluor_test_work_2026-01-21_09-06-17.326__hlg_1_idx_19_cnt_756_628_dst_25263.85_amp
                                        #   - Alm_1_fluor_test_work_2026-01-21_09-06-17.326__hlg_1_idx_19_cnt_756_628_dst_25263.85_flr_1
                                        #   - Base name: Alm_1_fluor_test_work_2026-01-21_09-06-17.326__hlg_1_idx_19_cnt_756_628_dst_25263.85_
                                        base_name = name[:-len(suffix)]
                                        
                                        # If new base name
                                        if base_name not in img_paths:
                                            # Create entry for new microalga
                                            img_paths[base_name] = {}
                                            
                                            # Set label
                                            img_paths[base_name]["class"] = class_label
                                            
                                        # Add channel path to corresponding microalga
                                        # Example: {base_name1: {"amp": path_amp, "fluor": path_fluor, ...}, base_name2: {...}}
                                        img_paths[base_name][suffix] = img

        
        # Clean the data, ensuring each microalga has all channels
        img_paths = self.clean_incomplete_data(img_paths)

        print("----")
        return img_paths            

    def clean_incomplete_data(self, data):
        
        clean_data = {}
        n_channels = len(config.IMG_SUFFIXES) + 1  # IMG_SUFFIXES + class
        
        
        for img_name, fields in data.items():
            if len(fields) == n_channels:
                clean_data[img_name] = fields
                
        print("Number of initial elements: ", len(data))
        print("Number of elements with all channels: ", len(clean_data))
        
        return clean_data
        
            
        
        
        
        
        
        
        
        
        
        
        