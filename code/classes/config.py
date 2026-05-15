# Prefix used for mapping the classes in this dataset
CLASS_PREFIXES = {"CH" : 0, # Chlorella
                "HA" : 1, # Haematococcus
                "SC" : 2
               }

# Name of each class
CLASS_NAMES = {"CH" : "Chlorella",
                "HA" : "Haematococcus",
                "SC" : "Scenedesmus"
               }

# 7 Channels per microalga
INITIAL_IMG_SUFFIXES = ["amp", "flr_1", "flr_2", "flr_3", "flu", "mask", "phase"] 
SELECTED_IMG_SUFFIXES = ["amp", "flr_1", "flr_2", "flr_3", "mask", "phase"] 

PIXEL_SIZE = 0.57971  # um per pixel

MODEL_PATH = "classes/saved_models/best_model.pth"


class Channels:
    MASK_AREA = "MASK_AREA"
    MASK_PERIMETER = "MASK_PERIMETER"
    MASK_CIRCULARITY = "MASK_CIRCULARITY"
    MASK_SOLIDITY = "MASK_SOLIDITY"
    MASK_ASPECTRATIO = "MASK_ASPECTRATIO"
    MEAN_FLUORESCENCE_FLU1 = "MEAN_FLUORESCENCE_FLU1"
    MEAN_FLUORESCENCE_FLU2 = "MEAN_FLUORESCENCE_FLU2"
    MEAN_FLUORESCENCE_FLU3 = "MEAN_FLUORESCENCE_FLU3"
    FLUORESCENT_AREA_RATIO_FLU1 = "FLUORESCENT_AREA_RATIO_FLU1"
    FLUORESCENT_AREA_RATIO_FLU2 = "FLUORESCENT_AREA_RATIO_FLU2"
    FLUORESCENT_AREA_RATIO_FLU3 = "FLUORESCENT_AREA_RATIO_FLU3"
    
SELECTED_FEATURES = [
    Channels.MASK_AREA,
    Channels.MASK_SOLIDITY,
    Channels.MASK_ASPECTRATIO,
    Channels.MEAN_FLUORESCENCE_FLU2,
    Channels.FLUORESCENT_AREA_RATIO_FLU2,
]