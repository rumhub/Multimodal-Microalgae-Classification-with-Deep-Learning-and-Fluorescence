from . import config
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split

class DataAnalysis:
    def __init__(self):
        pass
    
    '''
    @brief: Calculates and stores information about each microalga
    @paaram img_paths: Dictionary with the microalgae images
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
        #   x MASK_EXTENT               : Relation between real area and bounding box
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
            
            # Calculate mask contours
            mask_contours = self.calculate_mask_contours(mask)
            
            # Get mask info
            img_paths[img_name][config.Channels.MASK_AREA] = self.calculate_mask_area(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_PERIMETER] = self.calculate_mask_perimeter(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_CIRCULARITY] = self.calculate_mask_circularity(img_paths[img_name][config.Channels.MASK_AREA], img_paths[img_name][config.Channels.MASK_PERIMETER])
            img_paths[img_name][config.Channels.MASK_SOLIDITY] = self.calculate_mask_solidity(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_ASPECTRATIO] = self.calculate_mask_aspectratio(mask, mask_contours)
            img_paths[img_name][config.Channels.MASK_EXTENT] = self.calculate_mask_extent(mask, mask_contours)


            # print(f"Microalga: {img_name}")
            # print(f"Field: {field}")
        return img_paths
            
            
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
        try:
            # Transform to float
            area = float(area)
            perimeter = float(perimeter)
    
            # Validate input
            if perimeter <= 0 or area <= 0:
                return 0.0
            
            # Sources: 
            #   - https://math.stackexchange.com/questions/3496557/calculating-circularity
            #   - https://en.wikipedia.org/wiki/Roundness
            circularity = (perimeter ** 2) / (4 * np.pi * area)
    
            # Avoid NaN or infinite values
            if not np.isfinite(circularity):
                return 0.0
    
            return circularity
    
        except (TypeError, ValueError):
            return 0.0
    
    
    def calculate_mask_solidity(self, mask_img, contours):
        try:
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
    
        except Exception:
            return 0.0
    
    def calculate_mask_aspectratio(self, mask_img, contours=None):
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
            # Source: https://en.wikipedia.org/wiki/Aspect_ratio
            ar = min(w, h) / max(w, h)    # <= 1
    
            if not np.isfinite(ar):
                return 0.0
    
            return float(ar)
    
        except Exception:
            return 0.0
        
    def calculate_mask_extent(self, mask_img, contours):
        try:
            if mask_img is None:
                return 0.0
    
            area_px = cv2.countNonZero(mask_img)
            if area_px <= 0:
                return 0.0
    
            pts = cv2.findNonZero(mask_img)
            if pts is None:
                return 0.0
    
            x, y, w, h = cv2.boundingRect(pts)  # axis-aligned bbox in pixel grid
            box_area = float(w * h)
            if box_area <= 0:
                return 0.0
    
            extent = area_px / box_area  # <= 1 always
    
            if not np.isfinite(extent):
                return 0.0
    
            # Por seguridad numérica (no debería hacer falta, pero nunca sobra)
            if extent > 1.0:
                extent = 1.0
    
            return float(extent)
    
        except Exception:
            return 0.0
    
    
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