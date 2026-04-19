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
IMG_SUFFIXES = ["amp", "flr_1", "flr_2", "flr_3", "flu", "mask", "phase"] 

PIXEL_SIZE = 0.57971  # um per pixel

class Channels:
    MASK_AREA = "MASK_AREA"
    MASK_PERIMETER = "MASK_PERIMETER"
    MASK_CIRCULARITY = "MASK_CIRCULARITY"
    MASK_SOLIDITY = "MASK_SOLIDITY"
    MASK_ASPECTRATIO = "MASK_ASPECTRATIO"
    MASK_EXTENT = "MASK_EXTENT"