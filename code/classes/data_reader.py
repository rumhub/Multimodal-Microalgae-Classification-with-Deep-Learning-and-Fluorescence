from pathlib import Path
from . import config
import cv2

class DataReader:
    
    def __init__(self, path, generated_dir="../images/generated"):
        self.path = path
        self.generated_dir = Path(generated_dir)


    """
    @brief: Reads images and stores their paths + corresponding labels.

    Dataset structure expected:

        images/
            Ch/
                class_x/
                    sample_amp.png
                    sample_flu.png
                    sample_mask.png
                    sample_phase.png
            SCEN/
                class_x/
                    sample_amp.png
                    ...
            Haematococcus_verde1/
                class_x/
                    sample_amp.png
                    ...

    The real class is inferred from the main folder name:
        CH -> Chlorella
        HA -> Haematococcus
        SC -> Scenedesmus

    The folders named class_* are not used as labels, because they come from
    the previous classifier and may be incorrect

    @return: Dictionary with one entry per microalga
    """
    def read_data(self):
    
        print("--------- IMAGE READING ----------------")
    
        # Dictionary where all samples will be stored
        # Each key is the common base name of one microalga
        img_paths = {}
    
        # Get base path to read from
        base_dir = Path(self.path)
    
        print(f"Reading images from {base_dir.resolve()}\n")
        print("----")
    
        # Iterate over top-level folders:
        # Ch, Chlorella, SCEN, Haematococcus_verde1, etc.
        for base_class_dir in base_dir.iterdir():
    
            # Only look for folders
            if not base_class_dir.is_dir():
                continue
    
            # Get folder prefix to infer class
            class_dir_prefix = base_class_dir.name[:2].upper()
    
            # Ignore folders that are not one of the target classes
            # For example: generated/
            if class_dir_prefix not in config.CLASS_PREFIXES:
                continue
    
            # Get numeric label from top-level folder
            class_label = config.CLASS_PREFIXES[class_dir_prefix]
    
            print(f"Reading as {config.CLASS_NAMES[class_dir_prefix]} images from folder: {base_class_dir.name}")
    
            # ------------------------------------------------------------
            # Find class_* folders directly inside the top-level folder
            # ------------------------------------------------------------
            # The name of these folders is ignored as label
            # We only use them as containers of image files
            class_dirs = [
                d
                for d in base_class_dir.iterdir()
                if d.is_dir() and d.name.startswith("class")
            ]
    
            # If no class folders are found, skip this folder
            if len(class_dirs) == 0:
                print(f"[WARNING] No class folders found inside {base_class_dir}")
                continue
    
            # ------------------------------------------------------------
            # Read all image paths inside every class_* folder
            # ------------------------------------------------------------
            for class_dir in class_dirs:
    
                for img in class_dir.iterdir():
    
                    # Ignore possible subfolders or non-file entries
                    if not img.is_file():
                        continue
    
                    # Remove extension
                    # Example:
                    #   sample_amp.png -> sample_amp
                    name = img.stem
    
                    # ----------------------------------------------------
                    # Group image channels belonging to the same microalga
                    # ----------------------------------------------------
                    # Each image name ends with one known suffix:
                    #   amp, flr_1, flr_2, flr_3, flu, mask, phase
                    #
                    # The common microalga name is obtained by removing
                    # the suffix from the image name.
                    #
                    # Example:
                    #   sample_amp  -> base_name = sample_
                    #   sample_flu  -> base_name = sample_
                    # ----------------------------------------------------
                    for suffix in config.INITIAL_IMG_SUFFIXES:
    
                        if name.endswith(suffix):
                            
                            # base_name = name[:-len(suffix)]
                            base_name = f"{base_class_dir.name}_{class_dir.name}_{name[:-len(suffix)]}"
        
                            # Create new sample entry if needed
                            if base_name not in img_paths:
                                img_paths[base_name] = {}
        
                                # Assign real class using the top-level folder
                                img_paths[base_name]["class"] = class_label
        
                            # Store path of the current channel
                            img_paths[base_name][suffix] = img
    
        # Generate missing individual fluorescence channels from flu
        img_paths = self.generate_missing_fluorescence_channels_from_flu(img_paths)
    
        # Clean the data, ensuring each microalga has all channels
        img_paths = self.clean_incomplete_data(img_paths)
    
        print("----")
    
    
        # Print number of microalgae per class
        final_class_counts = {}
        
        for class_prefix in config.CLASS_PREFIXES:
            final_class_counts[class_prefix] = 0
    
        for fields in img_paths.values():
            class_label = fields.get("class")
    
            for class_prefix, label in config.CLASS_PREFIXES.items():
                if class_label == label:
                    final_class_counts[class_prefix] += 1
    
        for class_prefix, count in final_class_counts.items():
            print(f"{config.CLASS_NAMES[class_prefix]}: {count}")
            
        print("----")
    
        return img_paths

    def clean_incomplete_data(self, data):
        clean_data = {}
    
        required_fields = set(config.INITIAL_IMG_SUFFIXES)
        required_fields.add("class")
    
        for img_name, fields in data.items():
    
            if all(field in fields for field in required_fields):
                clean_data[img_name] = fields
    
        print("Number of initial elements: ", len(data))
        print("Number of elements with all channels: ", len(clean_data))
    
        return clean_data
        
            
    """
    @brief: Generates missing flr_1, flr_2 and flr_3 images from flu image

    The original dataset is not modified. Generated images are saved in:
        ../images/generated

    FLU composition:
        - Red channel   -> flr_1
        - Green channel -> flr_2
        - Blue channel  -> flr_3

    OpenCV reads images in BGR order:
        - channel 0 = Blue
        - channel 1 = Green
        - channel 2 = Red
    """
    def generate_missing_fluorescence_channels_from_flu(self, data):

        # Fluorescent channels
        flu_to_flr_map = {
            "flr_1": 2,
            "flr_2": 1,
            "flr_3": 0,
        }
    
        self.generated_dir.mkdir(parents=True, exist_ok=True)
    
        generated_count = 0
        reused_count = 0
        unrecovered_count = 0
    
        for img_name, fields in data.items():
    
            # If the sample does not have flu, nothing can be generated
            if "flu" not in fields:
                continue
    
            flu_img = None
    
            # For each fluorescent channel (flr_1, flr_2, flr_3)
            for flr_key, flu_channel_idx in flu_to_flr_map.items():
    
                # If the fluorescent channel does not exist in the current microalga, generate it 
                if flr_key not in fields:
    
                    # Read flu only when needed
                    if flu_img is None:
                        flu_img = cv2.imread(str(fields["flu"]), cv2.IMREAD_COLOR)
        
                        if flu_img is None:
                            print(f"[WARNING] Could not read flu image for {img_name}: {fields['flu']}")
                            unrecovered_count += 1
                            break
        
                    # Get image path
                    generated_path = self.generated_dir / f"{img_name}{flr_key}_from_flu.png"
        
                    # If it was already generated in a previous run, reuse it
                    if generated_path.exists():
                        fields[flr_key] = generated_path
                        reused_count += 1
                    else:
                        # Extract the corresponding channel from flu
                        generated_img = flu_img[:, :, flu_channel_idx]
            
                        success = cv2.imwrite(str(generated_path), generated_img)
            
                        if success:
                            fields[flr_key] = generated_path
                            generated_count += 1
                        else:
                            print(f"[WARNING] Could not save generated channel {flr_key} for {img_name}")
                            unrecovered_count += 1
    
        print(f"Generated fluorescence channels from flu: {generated_count}")
        print(f"Reused previously generated fluorescence channels: {reused_count}")
    
        if unrecovered_count > 0:
            print(f"Missing fluorescence channels that could not be generated: {unrecovered_count}")
    
        return data
        
        
        
        
        
        
        
        
        