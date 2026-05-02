from . import config
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.model_selection import train_test_split

class DataAnalysis:
    def __init__(self):
        self.channel_names = [value for name, value in vars(config.Channels).items() if not name.startswith("__")]
        self.channel_limits = None
        self.global_channel_limits = None
    
    
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
            fluor_1 = self.read_red_fluor_img(fields["flr_1"])
            fluor_2 = self.read_red_fluor_img(fields["flr_2"])
            fluor_3 = self.read_red_fluor_img(fields["flr_3"])
            fluor = self.read_red_fluor_img(fields["flu"])

            
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
        print(df.describe().T)
    
        for col in cols:
            values = df[col].dropna()
        
            if "AREA_RATIO" in col:
                high_ratio = np.mean(values >= 0.95)
                low_ratio = np.mean(values <= 0.01)
                print(f"{col}: high_ratio={high_ratio:.3f}, near_zero={low_ratio:.3f}")
            else:
                saturated = np.mean(values >= 250)
                near_zero = np.mean(values <= 5)
                print(f"{col}: saturated={saturated:.3f}, near_zero={near_zero:.3f}")
    
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
        
        self.print_summary(cols, img_paths)
    
        

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
        
    def read_red_fluor_img(self, img_path):
        # Read fluorescence image in color
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
        if img is None:
            return None
    
        # OpenCV uses BGR: channel 2 is red
        return img[:, :, 2]

    
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
    def show_histogram(self, data, title, xlabel="", ylabel=""):
            
        if len(data) == 0:
            return
        
        # Create figure
        plt.figure()
        plt.hist(data, bins=30)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
    
    
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
    @brief: Cleans the data, clean outliers and artifacts that are not microalgae
    @param data: Data to be cleaned
    
    '''
    def clean_data(self, data, debug = 0):
        
        # Get channel names
        channel_names = [value for name, value in vars(config.Channels).items() if not name.startswith("__")]

        # ---------------------------------------
        # ---------- Special filtering ----------
        # ---------------------------------------

        # ----- Remove data with mask area of 0  -----
        # Create a copy of the original data (list of img paths)
        img_names = list(data.keys())
        
        for img_name in img_names:
            # Avoid KeyError if img doesn't have channel channel_name
            value = data[img_name].get(config.Channels.MASK_AREA)

            # Remove negative and 0 values
            if value is None or value <= 0.001:
                del data[img_name]

        # ---------------------------------------
        # ---------- Common filtering ----------
        # ---------------------------------------
        
        # ---------- Remove data outside percentiles ----------
        imgs_to_remove = []
        # For each class or microalga type
        for current_class in config.CLASS_PREFIXES:
            
            class_data = {}
            
            # Get images per class
            for img_name, fields in data.items():
                
                if fields.get("class") == config.CLASS_PREFIXES[current_class]:
                    class_data[img_name] = fields
           
            # For each calculated channel (mask_area, mask_perimeter, etc)
            for channel_name in channel_names:

                # Get all the data for a determined channel
                channel_data = self.get_channel(class_data, channel_name)
                
                if channel_data is not None and len(channel_data) > 0:
                    
                    # Calculate percentiles per channel
                    p_low = np.percentile(channel_data, 5)
                    p_high = np.percentile(channel_data, 95)
            
                    # Filter data outside percentiles
                    img_names = list(class_data.keys())
                    for img_name in img_names:
                        # Avoid KeyError if img doesn't have channel channel_name
                        value = data[img_name].get(channel_name)
                        
                        # Mark data outside of percentiles to be removed
                        if img_name not in imgs_to_remove and value is not None:
                            if value < p_low or value > p_high:
                                imgs_to_remove.append(img_name)
            
           
        # Create a copy of the data before filtering
        if debug == 1:
            data_copy = copy.deepcopy(data)
       
        # Remove images
        for img in imgs_to_remove:
            if img in data:
                del data[img]
                
        # ------------------------------------------------------
        # -------- Show information after filtering ------------
        # ------------------------------------------------------
        if debug == 1:
            for current_class in config.CLASS_PREFIXES: # For each class or microalga type
                class_data = {}
                class_data_filtered = {}

                # Get images per class
                for img_name, fields in data.items():
                    if fields.get("class") == config.CLASS_PREFIXES[current_class]:
                        class_data_filtered[img_name] = fields
               
                # Get images per class
                for img_name, fields in data_copy.items():
                    if fields.get("class") == config.CLASS_PREFIXES[current_class]:
                        class_data[img_name] = fields 
               
                # For each calculated channel (mask_area, mask_perimeter, etc)
                for channel_name in channel_names:
    
                    # Get all the data for a determined channel
                    channel_data = self.get_channel(class_data, channel_name)
                    channel_data_filtered = self.get_channel(class_data_filtered, channel_name)
                    
                    if channel_data != None:
                        # Show general information about current channel
                        self.show_histogram(channel_data, f"Distribution of {channel_name} in class {config.CLASS_NAMES[current_class]} before filtering", channel_name, "Frecuency")
                        self.describe_channel(channel_data, f"------------ {channel_name} DESCRIPTION BEFORE FILTERING ------------")
                        
                        # Show general information about current channel
                        channel_data = self.get_channel(class_data, channel_name)
                        self.show_histogram(channel_data_filtered, f"Distribution of {channel_name} in class {config.CLASS_NAMES[current_class]} after filtering", channel_name, "Frecuency")
                        self.describe_channel(channel_data_filtered, f"------------ {channel_name} DESCRIPTION AFTER FILTERING ------------")
            
        return data
    
    
    '''
    @brief: Calculates limits for each class and channel
    @param data: Data to calculate limits from (should be training, not validation or test)
    @param p_low: low percentile to be used
    @param p_high: high percentile to be used
    '''
    def compute_limits_per_class(self, data, p_low=1, p_high=95):
        
        limits = {}
        
        # For each class
        for class_prefix, class_id in config.CLASS_PREFIXES.items():
            
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
                
                # Calculate percentiles
                low = np.percentile(channel_data, p_low)
                high = np.percentile(channel_data, p_high)
                
                # Store limits
                limits[class_id][channel] = {
                    "min": float(low),
                    "max": float(high)
                }
        
        self.channel_limits = limits
        
    
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
    
        
        # Manual FLu2 treshold
        mean_key = config.Channels.MEAN_FLUORESCENCE_FLU2
        ratio_key = config.Channels.FLUORESCENT_AREA_RATIO_FLU2
        
        if mean_key in global_limits:
            global_limits[mean_key]["min"] = max(global_limits[mean_key]["min"], 5.0)
            global_limits[mean_key]["max"] = min(global_limits[mean_key]["max"], 255.0)
        
        if ratio_key in global_limits:
            global_limits[ratio_key]["min"] = max(global_limits[ratio_key]["min"], 0.02)
            global_limits[ratio_key]["max"] = min(global_limits[ratio_key]["max"], 1.0)
        
        self.global_channel_limits = global_limits
    
    '''
    @brief: Cleans the data, clean outliers and artifacts that are not microalgae
    @param data: Data to be cleaned
    
    '''
    def global_filtering(self, data, debug = 0):
        
        # ---------------------------------------
        # ---------- Special filtering ----------
        # ---------------------------------------

        # ----- Remove data with mask area of 0  -----
        # Create a copy of the original data (list of img paths)
        img_names = list(data.keys())
        
        for img_name in img_names:
            # Avoid KeyError if img doesn't have channel channel_name
            value = data[img_name].get(config.Channels.MASK_AREA)

            # Remove negative and 0 values
            if value is None or value <= 0.001:
                del data[img_name]

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
                
                if value is not None and (value < p_low or value > p_high):
                    imgs_to_remove.add(img_name)
            
           
        # Create a copy of the data before filtering
        if debug == 1:
            data_copy = copy.deepcopy(data)
       
        # Remove images
        for img in imgs_to_remove:
            if img in data:
                del data[img]
                
        # ------------------------------------------------------
        # -------- Show information after filtering ------------
        # ------------------------------------------------------
        if debug == 1:
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
                        self.show_histogram(channel_data, f"Distribution of {channel_name} in class {config.CLASS_NAMES[class_prefix]} before filtering", channel_name, "Frecuency")
                        self.describe_channel(channel_data, f"------------ {channel_name} DESCRIPTION BEFORE FILTERING ------------")
                        
                        # Show general information about current channel
                        self.show_histogram(channel_data_filtered, f"Distribution of {channel_name} in class {config.CLASS_NAMES[class_prefix]} after filtering", channel_name, "Frecuency")
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