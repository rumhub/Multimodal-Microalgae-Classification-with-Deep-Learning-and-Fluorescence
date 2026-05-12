from . import config
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import pandas as pd
import random
import os
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

class DataAnalysis:
    def __init__(self):
        self.channel_names = [value for name, value in vars(config.Channels).items() if not name.startswith("__")]
        self.channel_limits = None
        self.global_channel_limits = None
    
    """
    @brief: Plots a grouped bar chart from fluorescence RGB channel statistics.
    
    @param summary: DataFrame returned by analyze_fluorescence_rgb_channels
    @param value_col: Column to plot, e.g. "mean", "std", "saturated_%"
    @param ylabel: Y-axis label
    @param title: Plot title
    @param save_path: Path where the figure will be saved. If None, it is only shown.
    """
    def plot_rgb_channel_summary(self, summary, value_col, ylabel, title, save_path=None):

    
        fluorescence_channels = ["FLR_1", "FLR_2", "FLR_3"]
        rgb_channels = ["Blue", "Green", "Red"]
    
        channel_colors = {
        "Blue": "#4C78A8",   # muted blue
        "Green": "#59A14F",  # muted green
        "Red": "#C44E52",    # muted red
        }
        
        x = np.arange(len(fluorescence_channels))
        width = 0.25
    
        plt.figure(figsize=(9, 5))
    
        for i, rgb_channel in enumerate(rgb_channels):
            values = []
    
            for fluorescence_channel in fluorescence_channels:
                row = summary[
                    (summary["fluorescence_channel"] == fluorescence_channel) &
                    (summary["rgb_channel"] == rgb_channel)
                ]
    
                if len(row) == 0:
                    values.append(0.0)
                else:
                    values.append(float(row[value_col].iloc[0]))
    
            plt.bar(
                x + (i - 1) * width,
                values,
                width,
                label=rgb_channel,
                color=channel_colors[rgb_channel]
            )
    
        plt.xticks(x, fluorescence_channels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
    
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
        plt.show()
        plt.close()
    
    """
    @brief: Analyzes BGR/RGB channel statistics for each fluorescence image type.

    This function helps decide which color channel is more informative for each
    fluorescence channel exported by Holodetect. It computes summary statistics
    and optionally saves bar plots for visual inspection.

    @param img_paths: Dictionary with the microalgae images
    @param save_dir: Directory where plots will be saved. If None, plots are only shown.

    @return: Pandas DataFrame with channel statistics
    """
    def analyze_fluorescence_rgb_channels(self, img_paths, save_dir=None):

    
        fluorescence_fields = {
            "flr_1": "FLR_1",
            "flr_2": "FLR_2",
            "flr_3": "FLR_3"
        }
    
        data = []
    
        for img_name, fields in img_paths.items():
    
            for field_key, field_name in fluorescence_fields.items():
    
                if field_key not in fields:
                    continue
    
                img = cv2.imread(str(fields[field_key]), cv2.IMREAD_COLOR)
    
                if img is None:
                    continue
    
                # OpenCV reads color images in BGR order.
                channels = {
                    "Blue": img[:, :, 0],
                    "Green": img[:, :, 1],
                    "Red": img[:, :, 2],
                }
    
                for channel_name, channel_img in channels.items():
    
                    data.append({
                        "image": img_name,
                        "fluorescence_channel": field_name,
                        "rgb_channel": channel_name,
                        "min": float(np.min(channel_img)),
                        "max": float(np.max(channel_img)),
                        "mean": float(np.mean(channel_img)),
                        "std": float(np.std(channel_img)),
                        "nonzero_%": float(np.mean(channel_img > 0) * 100),
                        "saturated_%": float(np.mean(channel_img >= 255) * 100),
                    })
    
        df = pd.DataFrame(data)
    
        if df.empty:
            print("No fluorescence images found.")
            return df
    
        summary = (
            df.groupby(["fluorescence_channel", "rgb_channel"])
            .agg({
                "min": "mean",
                "max": "mean",
                "mean": "mean",
                "std": "mean",
                "nonzero_%": "mean",
                "saturated_%": "mean",
            })
            .reset_index()
        )
    
        pd.set_option("display.max_columns", None)
        print(summary)
    
        # ------------------------------------------------------
        # Plot summary statistics
        # ------------------------------------------------------
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
        self.plot_rgb_channel_summary(
            summary,
            value_col="mean",
            ylabel="Mean intensity",
            title="Mean intensity by fluorescence channel and RGB channel",
            save_path=None if save_dir is None else os.path.join(save_dir, "fluorescence_rgb_mean_intensity.png")
        )
    
        self.plot_rgb_channel_summary(
            summary,
            value_col="std",
            ylabel="Standard deviation",
            title="Intensity variability by fluorescence channel and RGB channel",
            save_path=None if save_dir is None else os.path.join(save_dir, "fluorescence_rgb_std.png")
        )
    
        self.plot_rgb_channel_summary(
            summary,
            value_col="saturated_%",
            ylabel="Saturated pixels (%)",
            title="Saturation percentage by fluorescence channel and RGB channel",
            save_path=None if save_dir is None else os.path.join(save_dir, "fluorescence_rgb_saturation.png")
        )
    
        return summary
    
    """
    @brief: Verifies whether the composite fluorescence image matches the
            individual fluorescence channels

    The dataset contains three individual fluorescence images:
        - flr_1
        - flr_2
        - flr_3

    and one composite fluorescence image:
        - flu

    This function checks whether:
        - the red channel of flu matches the red channel of flr_1
        - the green channel of flu matches the red channel of flr_2
        - the blue channel of flu matches the red channel of flr_3

    OpenCV reads color images in BGR order, so:
        - channel 0 = Blue
        - channel 1 = Green
        - channel 2 = Red

    @param img_paths: Dictionary with the microalgae image paths
    @param num_samples: Number of random samples to verify
    @param seed: Random seed for reproducibility

    @return: True if all checked samples satisfy the expected relation, False otherwise
    """
    def verify_flu_composition(self, img_paths, num_samples=300, seed=42):
    
        # Convert dictionary items to a list so random sampling can be applied
        items = list(img_paths.items())
    
        # Avoid requesting more samples than available
        num_samples = min(num_samples, len(items))
    
        # Select random samples reproducibly
        rng = random.Random(seed)
        selected_items = rng.sample(items, num_samples)
    
        all_ok = True
    
        for img_name, fields in selected_items:
    
            # Read individual fluorescence images
            flr_1 = cv2.imread(str(fields["flr_1"]), cv2.IMREAD_COLOR)
            flr_2 = cv2.imread(str(fields["flr_2"]), cv2.IMREAD_COLOR)
            flr_3 = cv2.imread(str(fields["flr_3"]), cv2.IMREAD_COLOR)
    
            # Read composite fluorescence image
            flu = cv2.imread(str(fields["flu"]), cv2.IMREAD_COLOR)
    
            # Skip or fail if any image could not be read
            if flr_1 is None or flr_2 is None or flr_3 is None or flu is None:
                print(f"[ERROR] Could not read one or more fluorescence images for {img_name}")
                all_ok = False
                continue
    
            # Extract red channel from individual fluorescence images
            # Individual fluorescence images store their signal in the red channel
            flr_1_red = flr_1[:, :, 2]
            flr_2_red = flr_2[:, :, 2]
            flr_3_red = flr_3[:, :, 2]
    
            # Extract B, G and R channels from the composite fluorescence image
            flu_blue = flu[:, :, 0]
            flu_green = flu[:, :, 1]
            flu_red = flu[:, :, 2]
    
            # Compare the composition channels against the individual fluorescence channels
            red_matches = np.array_equal(flu_red, flr_1_red)
            green_matches = np.array_equal(flu_green, flr_2_red)
            blue_matches = np.array_equal(flu_blue, flr_3_red)
    
            if not (red_matches and green_matches and blue_matches):
                all_ok = False
                print(f"[MISMATCH] {img_name}")
                print(f"  FLU red   == FLR_1 red: {red_matches}")
                print(f"  FLU green == FLR_2 red: {green_matches}")
                print(f"  FLU blue  == FLR_3 red: {blue_matches}")
    
        if all_ok:
            print(f"All {num_samples} checked samples match the expected FLU composition")
        else:
            print("Some samples do not match the expected FLU composition")
    
        return all_ok
    
    '''
    @brief: Calculates and stores information about each microalga
    @param img_paths: Dictionary with the microalgae images
    '''
    def get_img_metrics(self, img_paths):
        
        # MASK METRICS
        # HOLODETECT USES: 
        #   x MASK_AREA                 : Size of the mask, number of white pixels
        #   - MASK_NUMBERSUBSEGMENT     : Number of objects connected in the mask
        #   - MASK_SUMMEDAREA           : Sum of areas from every object
        #   - MASK_MINENCCIRCDIAMETER   : Diameter of minimun circle that encapsulates the whole object
        #   x MASK_PERIMETER            : Perimeter of the mask
        #   x MASK_ASPECTRATIO          : Major_axis / Minor_axis
        #   - MASK_MAJOR_AXIS           : Major axis ellipse length
        #   - MASK_MINOR_AXIS           : Minor axis ellipse length
        #   x MASK_CIRCULARITY          : Circularity of the object
        #   - MASK_COMPACTNESS          : How close is to a compact form
        #   - MASK_ELONGATION           : How elongated the object is
        #   - MASK_RECT_AREA            : Area of the minimun bounding box that encloses the object
        #   - MASK_EXTENT               : Relation between real area and bounding box
        #   - MASK_EQUICIRCDIAMETER     : Diameter of the circle with the same area
        #   - MASK_LENGTH               : Maximum length of the object
        #   - MASK_THICKNESS            : Average thickness of the object
        #   - MASK_CENTER_POINT_X       : X coordinate of the centroid
        #   - MASK_CENTER_POINT_Y       : Y coordinate of the centroid
        #   - MASK_IS_CONV              : 1 -> Convex object, 0 -> has concavities
        #   x MASK_SOLIDITY             : Relation between area and area of the convex hull
        # ---------------------------------------------------------------------------------
        # Metrics used are the ones with and "x" instead of a "-".
        # At the moment, every metric used calculates the same value as holodetect, except for MASK_CIRCULARITY
        
        # img_name is the common name of the microalga, and field is each field from config.IMG_SUFFIXES
        # Example: img_name = chorella124514123, field
        for img_name, fields in img_paths.items():
            # Read mask channel
            mask = self.read_mask_img(fields["mask"])
            fluor_1 = self.read_fluor_img(fields["flr_1"])
            fluor_2 = self.read_fluor_img(fields["flr_2"])
            fluor_3 = self.read_fluor_img(fields["flr_3"])
            fluor = self.read_fluor_img(fields["flu"])

            # Calculate mask contours
            mask_contours = self.calculate_mask_contours(mask)
            
            # Get mask info
            img_paths[img_name][config.Channels.MASK_AREA] = self.calculate_mask_area(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_PERIMETER] = self.calculate_mask_perimeter(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_CIRCULARITY] = self.calculate_mask_circularity(img_paths[img_name][config.Channels.MASK_AREA], img_paths[img_name][config.Channels.MASK_PERIMETER])
            img_paths[img_name][config.Channels.MASK_SOLIDITY] = self.calculate_mask_solidity(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_ASPECTRATIO] = self.calculate_mask_aspectratio(mask)
            
            img_paths[img_name][config.Channels.MEAN_FLUORESCENCE_FLU1] = self.calculate_mean_fluorescence(fluor_1, mask)
            img_paths[img_name][config.Channels.MEAN_FLUORESCENCE_FLU2] = self.calculate_mean_fluorescence(fluor_2, mask)
            img_paths[img_name][config.Channels.MEAN_FLUORESCENCE_FLU3] = self.calculate_mean_fluorescence(fluor_3, mask)
            img_paths[img_name][config.Channels.MEAN_FLUORESCENCE_FLU] = self.calculate_mean_fluorescence(fluor, mask)
            
            img_paths[img_name][config.Channels.FLUORESCENT_AREA_RATIO_FLU1] = self.calculate_fluorescent_area_ratio(fluor_1)
            img_paths[img_name][config.Channels.FLUORESCENT_AREA_RATIO_FLU2] = self.calculate_fluorescent_area_ratio(fluor_2)
            img_paths[img_name][config.Channels.FLUORESCENT_AREA_RATIO_FLU3] = self.calculate_fluorescent_area_ratio(fluor_3)
            img_paths[img_name][config.Channels.FLUORESCENT_AREA_RATIO_FLU] = self.calculate_fluorescent_area_ratio(fluor)



            # print(f"Microalga: {img_name}")
            # print(f"Field: {field}")
        return img_paths
            
    
    '''
    @brief: Plot all channels of a microalga
    @param img_paths: Dictionary containing all microalgae
    @param idx: Number of the microalgae to show
    '''
    def plot_microalga_channels(self, img_paths, idx=0):
        
        # Get indexes
        keys = list(img_paths.keys())
        
        if idx >= len(keys):
            raise ValueError(f"Index {idx} out of range. Total microalgae: {len(keys)}")
        
        # Get micrcoalga name from its index
        name = keys[idx]
        microalga = img_paths[name]
        
        # Channels only (exclude class)
        channels = [ch for ch in config.IMG_SUFFIXES if ch != "class"]
    
        # Get microalga's class name
        class_name = list(config.CLASS_NAMES.values())[microalga["class"]]
    
        # Plot channels of current microalga
        fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4))
    
        if len(channels) == 1:
            axes = [axes]
    
        for ax, ch in zip(axes, channels):
            img = cv2.imread(str(microalga[ch]), cv2.IMREAD_UNCHANGED)
        
            if img is None:
                continue
        
            if len(img.shape) == 3:  # color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:  # grayscale
                ax.imshow(img, cmap="gray")
        
            ax.set_title(ch)
            ax.axis("off")
    
        plt.suptitle(f"{name} | Class: {class_name}")
        plt.tight_layout()
        plt.show()
    
            
    def plot_fluorescence_correlation(self, img_paths):
        cols = [
            config.Channels.MEAN_FLUORESCENCE_FLU1,
            config.Channels.MEAN_FLUORESCENCE_FLU2,
            config.Channels.MEAN_FLUORESCENCE_FLU3,
            config.Channels.MEAN_FLUORESCENCE_FLU,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU1,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU2,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU3,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU,
        ]
        self.plot_correlation(cols, img_paths, "Correlation matrix of fluorescence channels")
    
    
    def plot_mofologic_correlation(self, img_paths):
        cols = [
            config.Channels.MASK_AREA,
            config.Channels.MASK_PERIMETER,
            config.Channels.MASK_CIRCULARITY,
            config.Channels.MASK_SOLIDITY,
            config.Channels.MASK_ASPECTRATIO,
        ]
        self.plot_correlation(cols, img_paths, "Correlation matrix of morphological features")
    
        
    def plot_final_variables_correlation(self, img_paths):
        cols = [
            config.Channels.MASK_AREA,
            config.Channels.MASK_SOLIDITY,
            config.Channels.MASK_ASPECTRATIO,
            config.Channels.MEAN_FLUORESCENCE_FLU2,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU2
        ]
        self.plot_correlation(cols, img_paths, "Correlation matrix of final variables")
        
    
    def plot_correlation(self, cols, img_paths, title):
        data = []
        for _, fields in img_paths.items():
            row = {}
            for col in cols:
                if col in fields:
                    row[col] = fields[col]
            if len(row) == len(cols):
                data.append(row)
        
        df = pd.DataFrame(data)
        corr = df.corr()
        
        plt.figure(figsize=(7, 6))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
        plt.yticks(range(len(cols)), cols)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    
    def print_summary(self, cols, img_paths):
        data = []
    
        for _, fields in img_paths.items():
            row = {}
    
            for col in cols:
                if col in fields:
                    row[col] = fields[col]
    
            if len(row) == len(cols):
                data.append(row)
    
        df = pd.DataFrame(data)
    
        pd.set_option("display.max_columns", None)
    
        summary_df = df.describe().T
    
        # Add saturation percentage
        saturated_percentages = {}
    
        for col in cols:
            max_value = df[col].max()
    
            if max_value <= 1.0:
                saturation_value = 1.0
            else:
                saturation_value = 255.0
    
            saturated_percentages[col] = (df[col] >= saturation_value).mean() * 100
    
        summary_df["saturated_%"] = pd.Series(saturated_percentages)
    
        print(summary_df)
    
        return summary_df
    
    def print_fluorescence_summary(self, img_paths):
        cols = [
            config.Channels.MEAN_FLUORESCENCE_FLU1,
            config.Channels.MEAN_FLUORESCENCE_FLU2,
            config.Channels.MEAN_FLUORESCENCE_FLU3,
            config.Channels.MEAN_FLUORESCENCE_FLU,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU1,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU2,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU3,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU,
        ]
        
        return self.print_summary(cols, img_paths)
    
    def print_selected_variables_summary(self, img_paths):
        cols = config.SELECTED_FEATURES
        
        return self.print_summary(cols, img_paths)
    
        

    def split_train_val_test(self, data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    
        # Ensure correct format
        if abs(train_size + val_size + test_size - 1.0) > 1e-8:
            raise ValueError("train_size + val_size + test_size must be 1.0")

        # Get image names        
        img_names = list(data.keys())
        
        # Get labels
        labels = [data[img_name]["class"] for img_name in img_names]
    
        # Split train vs (validation + test)
        train_names, val_test_names = train_test_split(
            img_names,
            test_size=(1 - train_size),
            random_state=random_state,
            stratify=labels
        )
    
        # Get labels from validation+test set
        val_test_labels = [data[img_name]["class"] for img_name in val_test_names]
    
        # Second split, validation vs test
        val_ratio_in_temp = val_size / (val_size + test_size)
    
        val_names, test_names = train_test_split(
            val_test_names,
            test_size=(1 - val_ratio_in_temp),
            random_state=random_state,
            stratify=val_test_labels
        )
    
        # Get data from the image names of each split
        train_data = {img_name: data[img_name] for img_name in train_names}
        val_data = {img_name: data[img_name] for img_name in val_names}
        test_data = {img_name: data[img_name] for img_name in test_names}
    
        return train_data, val_data, test_data
        
            
    def read_mask_img(self, mask_path):
        # Read img as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
        # Convert image to binary, black and white
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return binary
    
    """
    @brief: Reads a fluorescence image as a single-channel grayscale image

    Fluorescence images may be stored as RGB/BGR visualizations, so reading in grayscale provides a
    single intensity image using all available channels
    """    
    def read_fluor_img(self, img_path):

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    
        if img is None:
            return None
    
        # Read red channel
        return img[:,:,2]


    
    def calculate_mask_contours(self, mask_img):
        # Extract contours from the binary mask
        # RETR_EXTERNAL ensures that only the outer boundary of the object is detected -> holes inside the object are ignored
        # CHAIN_APPROX_NONE keeps all contour points for maximum geometric precision.
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
        
    def calculate_mask_area(self, mask_img, contours):
    
        if not contours:
            return 0
    
        # Calculate the area of the contour
        area_px = cv2.contourArea(contours[0])
    
        # Convert area from pixel units to physical units (um2)
        # Value obtained from file summary.log (google drive)
        pixel_area = (config.PIXEL_SIZE)**2

        return area_px * pixel_area

    def calculate_mask_perimeter(self, mask_img, contours):

        if not contours:
            return 0
    
        # Calculate geometric perimeter in pixel units
        perimeter_px = cv2.arcLength(contours[0], True)
    
        # Convert from pixels to physical units (um2)
    
        return perimeter_px * config.PIXEL_SIZE
    
    def calculate_mask_circularity(self, area, perimeter):
        
        # Transform to float
        area = float(area)
        perimeter = float(perimeter)

        # Validate input
        if perimeter <= 0 or area <= 0:
            return 0.0
        
        # Sources: 
        #   - https://math.stackexchange.com/questions/3496557/calculating-circularity
        #   - https://en.wikipedia.org/wiki/Roundness
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        return circularity
    
    
    def calculate_mask_solidity(self, mask_img, contours):
        if not contours:
            return 0.0

        # Use the largest contour
        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        if area <= 0:
            return 0.0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return 0.0

        solidity = area / hull_area

        # Avoid NaN/inf
        if not np.isfinite(solidity):
            return 0.0

        return float(solidity)
    
    
    def calculate_mask_aspectratio(self, mask_img):
        try:
            if mask_img is None:
                return 0.0
    
            # Get white pixels coordinates
            pts = cv2.findNonZero(mask_img)
            if pts is None or len(pts) < 5:
                return 0.0
    
            # Get rotated rectangle of minimum area that fits those white pixels (oriented bounding box)
            rect = cv2.minAreaRect(pts)   
            w, h = rect[1]                  # Get only width and height of the rectangle
    
            if w <= 0 or h <= 0:
                return 0.0
    
            # The aspect ratio of a geometric shape is the ratio of its sizes in different dimensions. 
            # For example, the aspect ratio of a rectangle is the ratio of its longer side to its shorter 
            # side—the ratio of width to height,[1][2] when the rectangle is oriented as a "landscape".
            ar = min(w, h) / max(w, h)    # <= 1
    
            if not np.isfinite(ar):
                return 0.0
    
            return float(ar)
    
        except Exception:
            return 0.0

    '''
    @brief: Computes mean fluorescence inside the microalgae mask
    
    @param fluor_img: Fluorescence image (grayscale or single channel)
    @param mask_img: Binary mask of the microalga
    @param dilate: Whether to dilate the mask (recommended if fluorescence is more diffuse)
     param kernel_size: Size of dilation kernel
     
    @return: Mean fluorescence value
    '''
    def calculate_mean_fluorescence(self, fluor_img, mask_img, dilate=True, kernel_size=3):
    
        if fluor_img is None or mask_img is None:
            return 0.0
    
        # Ensure mask is binary
        mask = (mask_img > 0).astype(np.uint8)
    
        # Optional dilation to capture more fluorescence signal
        if dilate:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
    
        # Extract fluorescence values inside mask
        values = fluor_img[mask > 0]
    
        if values.size == 0:
            return 0.0
    
        mean_val = np.mean(values)
    
        # Avoid Nan
        if not np.isfinite(mean_val):
            return 0.0
    
        return float(mean_val)
 
    """
    @brief: Computes the proportion of fluorescent (bright) pixels in the image
    
    @param fluor_img: Fluorescence image (grayscale)
    @param threshold: Intensity threshold to consider a pixel as "fluorescent"
    
    @return: Ratio of bright pixels in [0,1]
    """
    def calculate_fluorescent_area_ratio(self, fluor_img, threshold=30):
       
       
       if fluor_img is None:
           return 0.0
       
       # Convert to numpy array if needed
       fluor = np.asarray(fluor_img)
       
       if fluor.size == 0:
           return 0.0
       
       # Count bright pixels
       bright_pixels = fluor > threshold
       
       if bright_pixels.size == 0:
           return 0.0
       
       # Compute ratio
       ratio = np.mean(bright_pixels)
       
       if not np.isfinite(ratio):
           return 0.0
       
       return float(ratio)
    
    '''
    @brief: Gets all microalgae data for a determined metric name
    @param img_paths: paths of the images to filter from
    @param channel_name: name of the metric to be used
    '''
    def get_channel(self, img_paths, channel_name):
        data = [img[channel_name] for img in img_paths.values() if channel_name in img]

        if len(data) == 0:
            return None
        
        return data
    
    '''
    @brief: Shows a histogram
    @param data: data to get the information from
    @param title: Title of the plot to be shown
    @param xlabel: Label of the X axis
    @param ylabel: Label of the Y axis
    '''
    def show_histogram(self, data, title, xlabel="", ylabel="", save_dir=None):
            
        if len(data) == 0:
            return
        
        # Create figure
        plt.figure()
        plt.hist(data, bins=30)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
            # Create safe filename from title
            filename = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title)
            filename = filename.strip("_") + ".png"
        
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
        plt.close()
    
    
    '''
    @brief: Show general information about a specific channel
    @param data: data to get the information from
    @param channel_name: Name of the channel to be used (just the name to print it out)
    @param percentiles: percentiles to be calculated and shown
    '''
    def describe_channel(self, data, title, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    
        if not data:
            return
    
        data = np.asarray(data, dtype=float)
        print(title)
        print("Number of microalgae: " + str(data.size))
        print("Minimum value: " + str(data.min()))
        print("Maximum value: " + str(data.max()))
        print("Mean value: " + str(data.mean()))
        print("Standard value: " + str(data.std()))
        print("Number of zeros: " + str(np.sum(data==0)))
        print("-- Percentiles --")
        for p in percentiles:
            print("Percentil " + str(p) + ": " + str(np.percentile(data, p)))


    '''
    @brief: Calculates and stores outliers
    @param data: Data to be analized
    '''
    def calculate_outliers(self, data):
        pass
            
    
    '''
    @brief: Calculates limits for each class and channel
    @param data: Data to calculate limits from (should be training, not validation or test)
    @param p_low: default low percentile to be used
    @param p_high: default high percentile to be used
    @param class_percentiles: optional dictionary with class-specific percentiles.
                              Example: {0: (1, 99), 1: (2, 98), 2: (5, 95)}
    '''
    def compute_limits_per_class(self, data, p_low=1, p_high=95, class_percentiles=None):
        
        limits = {}
        
        mean_key = config.Channels.MEAN_FLUORESCENCE_FLU2
        ratio_key = config.Channels.FLUORESCENT_AREA_RATIO_FLU2
        
        # For each class
        for class_prefix, class_id in config.CLASS_PREFIXES.items():
            
            # Select the percentile range for this class
            # If class_percentiles is provided, use the class-specific range.
            # Otherwise, use the default p_low and p_high values.
            if class_percentiles is not None and class_id in class_percentiles:
                current_p_low, current_p_high = class_percentiles[class_id]
            else:
                current_p_low, current_p_high = p_low, p_high
            
            # Get data for current class
            class_data = {}
            
            for img_name, fields in data.items():
                if fields.get("class") == class_id:
                    class_data[img_name] = fields
            
            # Skip if no data for this class
            if not class_data:
                continue
            
            limits[class_id] = {}
            
            # For each channel
            for channel in self.channel_names:
                
                # Get channel data
                channel_data = self.get_channel(class_data, channel)
                
                if channel_data is None or len(channel_data) == 0:
                    continue
                
                # Calculate class-specific percentiles
                low = np.percentile(channel_data, current_p_low)
                high = np.percentile(channel_data, current_p_high)
                
                # Store limits
                limits[class_id][channel] = {
                    "min": float(low),
                    "max": float(high)
                }
                
                # Fluorescence variables are saturated at the upper bound.
                # Therefore, they are only filtered by the lower percentile.
                if channel == mean_key:
                    limits[class_id][channel]["max"] = 255.0
                
                if channel == ratio_key:
                    limits[class_id][channel]["max"] = 1.0
        
        self.channel_limits = limits
        
    """
    @brief: Checks whether one sample passes the feature limits of a predicted class.
    
    The class-specific limits must have been previously computed using
    compute_limits_per_class().
    
    @param sample_features: Dictionary with the selected features of one sample
    @param predicted_class: Class predicted by the CNN
    
    @return: True if the sample is compatible with the predicted class limits,
             False otherwise
    """
    def passes_class_filter(self, sample_features, predicted_class):
    
        # Check that class limits have already been computed
        if self.channel_limits is None:
            raise ValueError("Class limits have not been computed. Call compute_limits_per_class() first.")
    
        # If there are no limits for the predicted class, reject the prediction
        if predicted_class not in self.channel_limits:
            return False
    
        # Get limits for the predicted class
        class_limits = self.channel_limits[predicted_class]
    
        # Check all features with limits for this class
        for channel_name, limits in class_limits.items():
    
            # If this feature is not present in the sample, skip it
            if channel_name not in sample_features:
                continue
    
            value = sample_features[channel_name]
    
            # Ignore missing values
            if value is None:
                continue
    
            # Get minimum and maximum allowed values for this feature
            min_value = limits["min"]
            max_value = limits["max"]
    
            # Reject the sample if the value is outside the class limits
            if value < min_value or value > max_value:
                return False
    
        # If all checked features are inside the limits, accept the sample
        return True
    
    '''
    @brief: Calculates global limits for each channel
    @param data: Data to calculate limits from (should be training, not validation or test)
    @param p_low: low percentile to be used
    @param p_high: high percentile to be used
    '''
    def compute_global_limits(self, data, p_low=1, p_high=99):
    
        global_limits = {}
    
        # For each channel
        for channel in self.channel_names:
    
            # Get channel data from all training samples
            channel_data = self.get_channel(data, channel)
    
            if channel_data is None or len(channel_data) == 0:
                continue
    
            # Calculate percentiles globally
            low = np.percentile(channel_data, p_low)
            high = np.percentile(channel_data, p_high)
    
            # Store limits
            global_limits[channel] = {
                "min": float(low),
                "max": float(high)
            }
    
        
        # Fluorescence variables are saturated at the upper bound
        # Therefore, they are only filtered by the lower percentile
        mean_key = config.Channels.MEAN_FLUORESCENCE_FLU2
        ratio_key = config.Channels.FLUORESCENT_AREA_RATIO_FLU2
        
        if mean_key in global_limits:
            global_limits[mean_key]["max"] = 255.0
    
        if ratio_key in global_limits:
            global_limits[ratio_key]["max"] = 1.0
            
        # MASK_AREA tends to saturate at its lower value, so its minimum limit is fixed to 0.0
        mask_key = config.Channels.MASK_AREA
        if mask_key in global_limits:
            global_limits[mask_key]["min"] = 0.0
        
        self.global_channel_limits = global_limits
    
    '''
    @brief: Cleans the data, clean outliers and artifacts that are not microalgae
    @param data: Data to be cleaned
    
    '''
    def global_filtering(self, data, debug = 0, save_dir = None):
        
        # Create a copy of the data before filtering
        if debug == 1:
            data_copy = copy.deepcopy(data)
        
        # ---------------------------------------
        # ---------- Special filtering ----------
        # ---------------------------------------

        # ----- Remove data with mask area of 0  -----
        # Create a copy of the original data (list of img paths)
        img_names = list(data.keys())
        
        fluorescence_channels = [
            config.Channels.MEAN_FLUORESCENCE_FLU2,
            config.Channels.FLUORESCENT_AREA_RATIO_FLU2,
        ]
        
        for img_name in img_names:
            # Avoid KeyError if img doesn't have channel channel_name
            value = data[img_name].get(config.Channels.MASK_AREA)

            # Remove negative and 0 values
            if value is None or value <= 0.001:
                del data[img_name]
                continue    # Make sure we only delete the img 1 time
                
            # ----- Remove data with low fluorescence -----
            for channel_name in fluorescence_channels:
                value = data[img_name].get(channel_name)
            
                if value is None or value < 0.05:
                    del data[img_name]
                    break   # Make sure we only delete the img 1 time
                    
        # ---------------------------------------
        # ---------- Common filtering ----------
        # ---------------------------------------
        
        # ---------- Remove data outside percentiles ----------
        imgs_to_remove = set()
    
        # For each calculated channel (mask_area, mask_perimeter, etc)
        for channel_name in self.channel_names:

            # Ensure channel exists
            if channel_name not in self.global_channel_limits:
                continue

            # Get percentiles per channel
            p_low = self.global_channel_limits[channel_name]["min"]
            p_high = self.global_channel_limits[channel_name]["max"]

            # Filter data outside percentiles
            for img_name, fields in data.items():
                # Avoid KeyError if img doesn't have channel channel_name
                value = fields.get(channel_name)
                
                # For fluorescence channels, only remove low values
                # High values are preserved because fluorescence may saturate
                if channel_name in fluorescence_channels:
                    if value < p_low:
                        imgs_to_remove.add(img_name)
                
                # For the rest of the channels, remove both low and high outliers
                else:
                    if value < p_low or value > p_high:
                        imgs_to_remove.add(img_name)
       
        # Remove images
        for img in imgs_to_remove:
            if img in data:
                del data[img]
                
        # ------------------------------------------------------
        # -------- Show information after filtering ------------
        # ------------------------------------------------------
        if debug == 1:
            print("Filtered elements: ", len(data_copy) - len(data))

            for class_prefix in config.CLASS_PREFIXES: # For each class or microalga type
                class_data = {}
                class_data_filtered = {}

                # Get images per class
                for img_name, fields in data.items():
                    if fields.get("class") == config.CLASS_PREFIXES[class_prefix]:
                        class_data_filtered[img_name] = fields
               
                # Get images per class
                for img_name, fields in data_copy.items():
                    if fields.get("class") == config.CLASS_PREFIXES[class_prefix]:
                        class_data[img_name] = fields 
               
                # For each calculated channel (mask_area, mask_perimeter, etc)
                for channel_name in self.channel_names:
    
                    # Get all the data for a determined channel
                    channel_data = self.get_channel(class_data, channel_name)
                    channel_data_filtered = self.get_channel(class_data_filtered, channel_name)
                    
                    if channel_data is not None and len(channel_data) > 0:
                        # Show general information about current channel
                        self.show_histogram(channel_data, f"Distribution of {channel_name} in class {config.CLASS_NAMES[class_prefix]} before filtering", channel_name, "Frecuency", save_dir)
                        self.describe_channel(channel_data, f"------------ {channel_name} DESCRIPTION BEFORE FILTERING ------------")
                        
                        # Show general information about current channel
                        self.show_histogram(channel_data_filtered, f"Distribution of {channel_name} in class {config.CLASS_NAMES[class_prefix]} after filtering", channel_name, "Frecuency", save_dir)
                        self.describe_channel(channel_data_filtered, f"------------ {channel_name} DESCRIPTION AFTER FILTERING ------------")
            
        return data
    
    
    """
    @brief: Splits each sample into image-channel data and selected feature data.
    
    @param img_paths: Dictionary containing image paths, class labels and calculated features.
    
    @return:
        image_data: Dictionary with image channel paths and class labels.
        feature_data: Dictionary with selected calculated features and class labels.
    """
    def split_data_from_features(self, img_paths):
    
        # Example: amp, flr_1, flr_2, flr_3, flu, mask, phase
        image_keys = set(config.IMG_SUFFIXES)
    
        # Selected calculated features used for filtering/tabular analysis.
        feature_keys = set(config.SELECTED_FEATURES)
    
        # Metadata to preserve in both outputs.
        metadata_keys = {"class"}
    
        image_data = {}
        feature_data = {}
    
        # Iterate over each microalga/sample
        for img_name, fields in img_paths.items():
    
            # Keep only image channels + metadata
            image_data[img_name] = {
                key: value
                for key, value in fields.items()
                if key in image_keys or key in metadata_keys
            }
    
            # Keep only selected calculated features + metadata
            feature_data[img_name] = {
                key: value
                for key, value in fields.items()
                if key in feature_keys or key in metadata_keys
            }
    
        return image_data, feature_data


    """
    @brief: Removes calculated features that are not selected.
            Image channels and metadata are kept.
    """
    def remove_unselected_features(self, img_paths):
    
        # Get all calculated feature names defined in config.Channels.
        # These are the fields that may be removed if they were not selected.
        all_features = {
            value
            for value in vars(config.Channels).values()
            if isinstance(value, str)
        }

        # Get the selected features    
        selected_features = set(config.SELECTED_FEATURES)
    
        # Output dictionary
        cleaned_data = {}
    
        # Iterate over every microalga/sample
        for img_name, fields in img_paths.items():
    
            # Keep fields that are NOT calculated features (amp, flr_1, flr_2, ..)
            #       AND calculated features that ARE selected
            cleaned_data[img_name] = {
                key: value
                for key, value in fields.items()
                if key not in all_features or key in selected_features
            }
    
        return cleaned_data
    
    """
    @brief: Forces all classes to have the same number of samples by undersampling
            the majority classes
    """
    def balance_classes_to_min_count(self, data, seed=42, debug=True):
        data_by_class = defaultdict(list)
        
        for img_name, fields in data.items():
            class_label = fields["class"]
            data_by_class[class_label].append((img_name, fields))
        
        if len(data_by_class) == 0:
            return {}
        
        min_count = min(len(samples) for samples in data_by_class.values())
        
        if debug:
            print("\n--------- CLASS BALANCING ----------------")
            print("Original distribution:")
            for class_label, samples in sorted(data_by_class.items()):
                print(f"Class {class_label}: {len(samples)} samples")
        
            print(f"\nBalancing all classes to {min_count} samples.")
        
        balanced_data = {}
        
        random.seed(seed)
        
        for class_label, samples in data_by_class.items():
            selected_samples = random.sample(samples, min_count)
        
            for img_name, fields in selected_samples:
                balanced_data[img_name] = fields
        
        if debug:
            balanced_by_class = defaultdict(int)
        
            for fields in balanced_data.values():
                balanced_by_class[fields["class"]] += 1
        
            print("\nBalanced distribution:")
            for class_label, count in sorted(balanced_by_class.items()):
                print(f"Class {class_label}: {count} samples")
        
            print("------------------------------------------\n")
        
        return balanced_data