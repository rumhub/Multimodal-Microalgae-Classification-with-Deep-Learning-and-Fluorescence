import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
Class that handles the CNN model
"""
class Model:

    """
    @brief: Initializes the CNN model

    @param selected_channels: List of image channels that will be used as input to the CNN.
                              For example: amp, phase, flr_2, mask, etc.

    @param num_classes: Number of classes to predict. 

    @param device : Device where the model will be trained. If None, the code uses
                    GPU when available and CPU otherwise.
    """
    def __init__(
        self,
        selected_channels,
        num_classes,
        device=None
    ):
        # Input channels
        self.selected_channels = selected_channels
        
        # Number of output classes
        self.num_classes = num_classes

        # DataLoaders are initialized later when the data is provided
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Select computation device: GPU if available, otherwise CPU
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build the CNN model and move it to the selected device
        self.model = self._build_model().to(self.device)
        
        # Loss function for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        

    """
    @brief: Creates dataloaders from train, validation and test dictionaries.

    @param train_images: Training image paths dictionary
    @param val_images: Validation image paths dictionary
    @param test_images: Test image paths dictionary
    """
    def read_data(self, train_images, val_images, test_images, batch_size=32, num_workers=2):

        # Create dataloaders
        self.train_loader = self._create_loader(train_images, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self._create_loader(val_images, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self._create_loader(test_images, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    """
    @brief: Trains the CNN model. If validation data is available, also loads the best model according to validation loss
            at the end of training
            
    @param num_epochs : Number of complete passes over the training dataset

    @param learning_rate: Step size used by the optimizer to update the model weights

    """
    def train(self, num_epochs=30, learning_rate=1e-4):
        
        # Check that the training DataLoader has already been created.
        # If not, the model cannot be trained.
        if self.train_loader is None:
            raise ValueError("Training data has not been loaded. Call read_data() first.")
    
        # Initilaize metrics of the best model
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = -1
        best_model_state = None
    
        # Optimizer used to update the CNN weights during training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    
        # Main training loop. Each iteration corresponds to one full pass
        # over the training dataset.
        for epoch in range(num_epochs):
            
            # Train the model for one epoch and obtain training loss and accuracy
            train_loss, train_acc = self._train_one_epoch(optimizer)
    
            # If a validation DataLoader exists, evaluate the model after
            # each training epoch.
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(split="val")
    
                # Check whether the current model is better than all previous ones, based on
                # validation loss
                if val_loss < best_val_loss:
                    
                    # Update the best model metrics
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
    
                    # Save a copy of the current model weights
                    #
                    # detach(): removes tensors from the computation graph
                    # cpu(): stores the copy in CPU memory instead of GPU memory
                    # clone(): creates an independent copy, avoiding references
                    #          to weights that will keep changing during training
                    best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    
                    # Text marker used only for printing
                    best_marker = " <-- best"
                else:
                    # No marker is printed if the current epoch is not the best one
                    best_marker = ""
    
                # Print training and validation metrics for the current epoch
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
                    f"{best_marker}"
                )
    
            else:
                # If there is no validation set, only training metrics are printed
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}"
                )
    
        # After training, restore the model weights from the epoch with the
        # lowest validation loss
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
            # Move the restored model back to the selected device
            # This is necessary because the saved copy was stored on CPU
            self.model.to(self.device)
    
            # Print final information about the restored best model
            print(
                f"\nBest model restored from epoch {best_epoch} "
                f"with Val loss: {best_val_loss:.4f} | Val acc: {best_val_acc:.4f}"
            )

    """
    @brief: Evaluates the model on a selected dataset split

    @param split: Dataset split to evaluate. It can be "train", "val" or "test"
    @return: Average loss and accuracy on the selected split
    """
    def evaluate(self, split="test"):

        # Select the corresponding DataLoader depending on the spli
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError("split must be 'train', 'val' or 'test'.")

        # Check that the selected DataLoader has been created
        if loader is None:
            raise ValueError(f"{split} data has not been loaded.")

        # Set the model to evaluation mode
        # This disables training-specific behavior such as dropout and changes
        # batch normalization to use learned statistics instead of batch statistics
        self.model.eval()

        # Accumulated loss over all evaluated samples
        total_loss = 0.0
        
        # Number of correctly classified samples
        correct = 0
        
        # Total number of evaluated samples
        total = 0

        # Disable gradient computation
        # During evaluation we do not update the model weights, so gradients
        # are not needed. This reduces memory usage and speeds up computation
        with torch.no_grad():
            
            # Iterate over all batches in the selected DataLoader
            for x, y in loader:
                
                # Move input images and labels to the selected device (GPU or CPU)
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                logits = self.model(x)
                
                # Compute the loss for the current batch
                loss = self.criterion(logits, y)

                # Accumulate the total loss
                #
                # loss.item() gives the average loss of the current batch
                # Multiplying by x.size(0) converts it into total batch loss,
                # so that later we can compute the correct average over all samples
                total_loss += loss.item() * x.size(0)

                # Get the predicted class for each sample
                # torch.argmax(logits, dim=1) returns the index of the class with
                # the highest score for each image in the batch
                preds = torch.argmax(logits, dim=1)
                
                # Count how many predictions are equal to the real labels
                correct += (preds == y).sum().item()
                
                # Count how many samples have been evaluated
                total += y.size(0)

        # Compute the average loss over all evaluated samples
        avg_loss = total_loss / total
        
        # Compute accuracy as number of correct predictions / total number of samples
        acc = correct / total

        return avg_loss, acc


    """
    Predicts the class of one sample.

    @param sample: Dictionary with image paths for the selected channels.
    @return: Predicted class index and confidence score.
    """
    def predict(self, sample):
        
        # Set the model to evaluation mode
        self.model.eval()

        # Load the sample and convert it into a tensor with the expected shape
        #
        # self._load_sample(sample) returns a tensor with shape:
        #     [channels, height, width]
        x = self._load_sample(sample)
        
        # Add a batch dimension because PyTorch models expect inputs with shape:
        #     [batch_size, channels, height, width]
        #
        # Since we are predicting only one sample, batch_size = 1
        # Then move the tensor to the same device as the model
        x = x.unsqueeze(0).to(self.device)

        # Disable gradient computation
        # During prediction we do not update the model weights, so gradients
        # are not needed. This reduces memory usage and speeds up inference
        with torch.no_grad():
            
            # Forward pass
            logits = self.model(x)
            
            # Convert logits into probabilities using softmax
            # Each value will be between 0 and 1, and all probabilities
            # for the sample will sum to 1
            probs = torch.softmax(logits, dim=1)
            
            # Get the highest probability and its corresponding class index
            confidence, predicted_class = torch.max(probs, dim=1)

        # Convert PyTorch tensors to standard Python values
        #
        # predicted_class.item() extracts the scalar value from the tensor
        # confidence.item() extracts the scalar probability from the tensor
        return int(predicted_class.item()), float(confidence.item())


    """
    @brief: Saves the trained model weights
    @param output_path: Path where the model weights will be saved
    """
    def save(self, output_path):
        
        # Save only the model parameters, not the full Python model object
        #
        # self.model.state_dict() contains the learned weights and biases
        # of the neural network
        torch.save(self.model.state_dict(), output_path)


    """
    @brief: Loads trained model weights
    @param model_path: Path to the saved model weights
    """
    def load(self, model_path):

        # Load the saved model parameters from disk
        #
        # map_location=self.device ensures that the weights are loaded onto
        # the correct device, either GPU or CPU
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Move the model to the selected device
        #
        # This ensures that the model and input tensors are on the same device
        # during evaluation or prediction
        self.model.to(self.device)


    """
    @brief: Trains the model for one epoch, (one pass over all batches 
                                             in the training set)
    
    @param optimizer: Optimizer used to update the CNN weights during training

    @return: Average training loss and training accuracy for the epoch
    """
    def _train_one_epoch(self, optimizer):

        # Set the model to training mode
        # This enables training-specific behavior in layers such as Dropout
        # and BatchNorm
        self.model.train()

        # Accumulated loss over all training samples
        total_loss = 0.0
        
        # Number of correctly classified samples
        correct = 0
        
        # Total number of processed samples
        total = 0

        # Iterate over all batches in the training DataLoader
        for x, y in self.train_loader:
            
            # Move input images and labels to the selected device (GPU or CPU)
            x = x.to(self.device)
            y = y.to(self.device)

            # Reset gradients from the previous batch
            #
            # PyTorch accumulates gradients by default, so this must be done
            # before computing the gradients for the current batch
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(x)
            
            # Compute the loss between the predicted logits and the true labels
            # CrossEntropyLoss expects logits directly, so no softmax is applied here
            loss = self.criterion(logits, y)

            # Backward pass
            # Computes the gradients of the loss with respect to the model parameters
            loss.backward()
            
            # Update the model parameters using the computed gradients
            optimizer.step()

            # Accumulate the total loss
            #
            # loss.item() is the average loss of the current batch.
            # Multiplying by x.size(0) converts it into total batch loss
            total_loss += loss.item() * x.size(0)

            # Get the predicted class for each sample
            # The predicted class is the index with the highest logit
            preds = torch.argmax(logits, dim=1)
            
            # Count correct predictions in the current batch
            correct += (preds == y).sum().item()
            
            # Count processed samples in the current batch
            total += y.size(0)

        # Compute average loss over all training samples
        avg_loss = total_loss / total
        
        # Compute training accuracy
        acc = correct / total

        return avg_loss, acc


    """
    @brief: Creates a PyTorch DataLoader from the given data

    @param data: Dictionary containing the samples to load
    
    @param shuffle: Whether the samples should be shuffled
                    Usually True for training and False for validation/test
               
    @param batch_size: Number of samples processed at the same time during training

    @param num_workers : Number of subprocesses used by the DataLoader to load data

    @return: PyTorch DataLoader
    """
    def _create_loader(self, data, shuffle, batch_size, num_workers=2):

        # Create a custom Dataset object
        #
        # The Dataset defines how each sample is accessed and loaded
        # In this case, _MicroalgaeDataset receives:
        #
        # - data: dictionary with all sample information
        # - selected_channels: channels that will be used as model input
        # - load_fn: function used to load one sample from disk
        dataset = _MicroalgaeDataset(
            data=data,
            selected_channels=self.selected_channels,
            load_fn=self._load_sample
        )

        # Create and return a PyTorch DataLoader
        #
        # The DataLoader groups samples into batches and handles iteration
        # during training, validation or testing
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            
            # If CUDA is available, use pinned memory
            #
            # This can speed up the transfer of batches from CPU RAM to GPU memory
            pin_memory=torch.cuda.is_available()
        )



    """
    @brief: Loads one sample and returns it as a PyTorch tensor

    Each selected image channel is loaded as a grayscale image, normalized
    to the range [0, 1], and then stacked into a multi-channel tensor

    @param fields: Dictionary containing the image paths for one sample
                   Each selected channel must be a key in this dictionary

    @return: Tensor with shape [C, H, W], where:
             C = number of selected channels
             H = image height
             W = image width
    """
    def _load_sample(self, fields):
        
        # List where each loaded channel image will be stored
        images = []

        # Load each selected channel for the current sample
        for channel in self.selected_channels:
            
            # Get the image path corresponding to the current channel
            img_path = fields[channel]


            # Read the image in grayscale mode
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # Check that the image was correctly read
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")

            # Convert the image to float32 and normalize pixel values to the range [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add the normalized channel image to the list
            images.append(img)

        # Stack all loaded channel images into a single NumPy array
        #
        # Before stacking:
        #   images = list of arrays with shape [H, W]
        #
        # After stacking with axis=-1:
        #   x has shape [H, W, C]
        x = np.stack(images, axis=-1)
        
        # Convert the NumPy array to a PyTorch tensor and reorder dimensions
        #
        # PyTorch convolutional networks expect images with shape:
        #   [C, H, W]
        #
        # But x currently has shape:
        #   [H, W, C]
        #
        # permute(2, 0, 1) changes:
        #   [H, W, C] -> [C, H, W]
        x = torch.from_numpy(x).permute(2, 0, 1).float()

        return x


    """
    @brief: Builds the CNN architecture

    @return: PyTorch neural network model
    """
    def _build_model(self):

        # Number of input channels of the CNN
        in_channels = len(self.selected_channels)

        # Define the CNN architecture as a sequence of layers
        model = nn.Sequential(
            
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        return model

"""
Internal dataset used only by Model.

This class adapts our dictionary structure to the format expected by PyTorch.
Each item returned by this dataset is:
    x -> image tensor with the selected channels
    y -> class label
"""
class _MicroalgaeDataset(Dataset):

    """
    @param data: Dictionary containing the samples.
                 Example:
                 {
                     "sample_001": {
                         "amp": path_to_amp,
                         "phase": path_to_phase,
                         "flr_2": path_to_flr2,
                         "class": 0
                     },
                     ...
                 }

    @param selected_channels: List of image channels used as input to the model.
                              Example: ["amp", "phase", "flr_2"]

    @param load_fn: Function used to load one sample from disk and convert it
                    into a tensor. In your case, this will usually be
                    self._load_sample from the Model class.
    """
    def __init__(self, data, selected_channels, load_fn):
        
        # Store input data
        self.data = data
        
        # Store number of channels to be used by the model
        self.selected_channels = selected_channels
        
        # Store the function that lods aand preprocesses one sample
        self.load_fn = load_fn
        
        # Store sample names in a list so PyTorch can access by index
        self.sample_names = list(data.keys())

    """
    @brief: Returns the number of samples in the dataset.

    PyTorch uses this to know how many samples are available.
    """
    def __len__(self):
        return len(self.sample_names)

    """
    @brief: Returns one sample from the dataset.

    @param idx: Index of the sample to load

    @return:
        x: image tensor used as input to the CNN
        y: class label as a tensor
    """
    def __getitem__(self, idx):
        
        # Get the sample name corresponding to this index
        sample_name = self.sample_names[idx]
        
        # Get all fields/channels of this microalga (aamp, phase, fl2, etc)
        fields = self.data[sample_name]

        # Load selected microalgaaa channels and convert them into a tensor
        x = self.load_fn(fields)
        
        # Get the class label and convert it to a PyTorch tensor
        # dtype=torch.long is required by CrossEntropyLoss
        y = torch.tensor(fields["class"], dtype=torch.long)

        return x, y