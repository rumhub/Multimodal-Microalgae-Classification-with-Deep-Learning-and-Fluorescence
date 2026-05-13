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
        
        self.training_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        }
            

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
    def train(self, num_epochs=30, learning_rate=1e-4, patience=40):
        
        # Check that the training DataLoader has already been created.
        # If not, the model cannot be trained.
        if self.train_loader is None:
            raise ValueError("Training data has not been loaded. Call read_data() first.")
    
        # Initilaize metrics of the best model
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = -1
        best_model_state = None
        epochs_without_improvement = 0
    
        # Optimizer used to update the CNN weights during training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)

    
        # Main training loop. Each iteration corresponds to one full pass
        # over the training dataset.
        for epoch in range(num_epochs):
            
            # Train the model for one epoch and obtain training loss and accuracy
            train_loss, train_acc = self._train_one_epoch(optimizer)
            
            # Store training metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)

            # If a validation DataLoader exists, evaluate the model after
            # each training epoch.
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(split="val")
                
                # Store validation metrics
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)
    
                # Check whether the current model is better than all previous ones, based on
                # validation loss
                if val_loss < best_val_loss:
                    
                    # Update the best model metrics
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
    
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
                    
                    # Count this epoch as not improvement
                    epochs_without_improvement += 1
    
                # Print training and validation metrics for the current epoch
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
                    f"{best_marker}"
                )
                
                if epochs_without_improvement >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch} | "
                        f"Val loss: {best_val_loss:.4f} | Val acc: {best_val_acc:.4f}"
                    )
                    break
    
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


    def plot_training_curves(self, save_dir=None):
        """
        @brief: Plots training and validation loss/accuracy curves.
    
        @param save_dir: Directory where the plots will be saved.
                         If None, plots are only shown.
        """
    
        import os
        import matplotlib.pyplot as plt
    
        if len(self.training_history["train_loss"]) == 0:
            print("No training history available. Train the model first.")
            return
    
        epochs = np.arange(1, len(self.training_history["train_loss"]) + 1)
    
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
        best_epoch = None
    
        if len(self.training_history["val_loss"]) > 0:
            best_epoch = int(np.argmin(self.training_history["val_loss"])) + 1
            best_val_loss = self.training_history["val_loss"][best_epoch - 1]
    
        # -------------------------
        # Loss curve
        # -------------------------
        plt.figure(figsize=(8, 5))
    
        plt.plot(
            epochs,
            self.training_history["train_loss"],
            label="Train loss",
            linewidth=1.8
        )
    
        if len(self.training_history["val_loss"]) > 0:
            plt.plot(
                epochs,
                self.training_history["val_loss"],
                label="Validation loss",
                linewidth=1.8
            )
    
            plt.axvline(
                best_epoch,
                linestyle="--",
                linewidth=1.2,
                color="gray",
                label=f"Best epoch: {best_epoch}"
            )
    
            plt.scatter(
                best_epoch,
                best_val_loss,
                color="black",
                s=30,
                zorder=3
            )
    
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, "training_validation_loss.png"),
                dpi=300,
                bbox_inches="tight"
            )
    
        plt.show()
        plt.close()
    
        # -------------------------
        # Accuracy curve
        # -------------------------
        plt.figure(figsize=(8, 5))
    
        plt.plot(
            epochs,
            self.training_history["train_acc"],
            label="Train accuracy",
            linewidth=1.8
        )
    
        if len(self.training_history["val_acc"]) > 0:
            plt.plot(
                epochs,
                self.training_history["val_acc"],
                label="Validation accuracy",
                linewidth=1.8
            )
    
            if best_epoch is not None:
                plt.axvline(
                    best_epoch,
                    linestyle="--",
                    linewidth=1.2,
                    color="gray",
                    label=f"Best epoch: {best_epoch}"
                )
    
                plt.scatter(
                    best_epoch,
                    self.training_history["val_acc"][best_epoch - 1],
                    color="black",
                    s=30,
                    zorder=3
                )
    
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, "training_validation_accuracy.png"),
                dpi=300,
                bbox_inches="tight"
            )
    
        plt.show()
        plt.close()
        
    """
    @brief: Evaluates the model on a selected dataset split

    @param split: Dataset split to evaluate. It can be "train", "val" or "test"
    @return: Average loss and accuracy on the selected split
    """
    def evaluate(self, split="test"):

        # Select the corresponding DataLoader depending on the spli
        loader = self._get_loader(split)

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
            for x, y, _ in loader:
                
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
    @brief: Predicts all samples in a dictionary

    @param data: Dictionary containing the samples

    @return: Dictionary with true class, predicted class and confidence
    """
    def predict_dataset(self, data):

    
        results = {}
    
        for sample_name, fields in data.items():
    
            predicted_class, confidence = self.predict(fields)
            true_class = fields["class"]
    
            results[sample_name] = {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
    
        return results


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
        for x, y, _ in self.train_loader:
            
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
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes)
        )

        return model


    """
    @brief: Computes one confidence threshold per predicted class using a selected split

    @param split: Dataset split used to compute the thresholds. Usually "val"
    @param min_accepted_ratio: Minimum fraction of predictions that must be accepted
                               for each predicted class
    @param thresholds: Candidate thresholds to evaluate
    @param verbose: If True, prints the threshold selection summary

    @return: Dictionary with one confidence threshold per class
    """
    def compute_class_confidence_thresholds(self, split="val", min_accepted_ratio=0.8, 
                                            thresholds=None, debug=True):
        
        # Set deefault thresholds if none are given
        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.01)
    
        # Get predictions from a data split (training, validation, test)
        results = self.predict_loader(split=split)
    
        class_thresholds = {
            class_idx: 1.0
            for class_idx in range(self.num_classes)
        }
    
        threshold_summary = {}
    
        if debug:
            print("\n--------- CONFIDENCE THRESHOLD SEARCH ----------------")
    
        # Calculate threshold for each class
        for class_idx in range(self.num_classes):
    
            # Get predictions for the current class
            class_predictions= [
                result
                for result in results
                if result["predicted_class"] == class_idx
            ]
    
            num_predictions = len(class_predictions)
    
            if num_predictions == 0:
                threshold_summary[class_idx] = {
                    "num_predictions": 0,
                    "threshold": 1.0,
                    "accepted": 0,
                    "accepted_ratio": 0.0,
                    "accepted_accuracy": 0.0
                }
    
                if debug:
                    print(
                        f"Class {class_idx} | "
                        f"Predicted samples: 0 | "
                        f"Threshold: 1.00 | "
                        f"Coverage: 0.0000 | "
                        f"Accepted acc: 0.0000"
                    )
    
                continue
    
            best_threshold = 0.0
            best_accuracy = -1.0
            best_accepted_ratio = 0.0
            best_num_accepted = 0
    
            # Search through all possible thresholds
            for threshold in thresholds:
    
                # Get accepted results for this threshold
                accepted_results = [
                    result
                    for result in class_predictions
                    if result["confidence"] >= threshold
                ]
    
                num_accepted = len(accepted_results)
                accepted_ratio = num_accepted / num_predictions
    
                if accepted_ratio < min_accepted_ratio:
                    continue
    
                if num_accepted == 0:
                    continue
    
                # Calculate accuracy on current accepted results by this threshold
                accuracy = sum(
                    result["predicted_class"] == result["true_class"]
                    for result in accepted_results
                ) / num_accepted
    
                if (accuracy > best_accuracy or
                    (accuracy == best_accuracy and accepted_ratio > best_accepted_ratio)):
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_accepted_ratio = accepted_ratio
                    best_num_accepted = num_accepted
    
            # Store best threshold for class class_idx
            class_thresholds[class_idx] = float(best_threshold)
    
            if debug:
                print(
                    f"Class {class_idx} | "
                    f"Predicted samples: {num_predictions} | "
                    f"Threshold: {best_threshold:.2f} | "
                    f"Accepted: {best_num_accepted}/{num_predictions} | "
                    f"Coverage: {best_accepted_ratio:.4f} | "
                    f"Accepted acc: {best_accuracy:.4f}"
                )
    
        if debug:
            print("------------------------------------------------------\n")
    
        return class_thresholds



    """
    @brief: Predicts all samples from a selected DataLoader
    
    If class_thresholds is provided, class-specific confidence filtering is
    also applied.
    
    @param split: Dataset split to predict. It can be "train", "val" or "test"
    @param class_thresholds: Optional dictionary with one confidence threshold per class
    
    @return: List of dictionaries with prediction results
    """
    def predict_loader(self, split="test", class_thresholds=None):
    
        # Get the DataLoader corresponding to the selected split
        loader = self._get_loader(split)
    
        # Set model to evaluation mode
        self.model.eval()
    
        # List where prediction results will be stored
        results = []
    
        # Disable gradient computation
        with torch.no_grad():
    
            # Iterate over all batches in the selected DataLoader
            for x, y, sample_names in loader:
    
                # Move batch to selected device
                x = x.to(self.device)
                y = y.to(self.device)
    
                # Forward pass
                logits = self.model(x)
    
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=1)
    
                # Get confidence and predicted class for each sample
                confidences, predicted_classes = torch.max(probs, dim=1)
    
                # Store results sample by sample
                for sample_name, true_class, predicted_class, confidence in zip(
                    sample_names,
                    y.cpu(),
                    predicted_classes.cpu(),
                    confidences.cpu()
                ):
                    # Convert from PyTorch scalar tensors to Python values
                    true_class = int(true_class.item())
                    predicted_class = int(predicted_class.item())
                    confidence = float(confidence.item())
    
                    # Store the results of the prediction
                    result = {
                        "sample_name": sample_name,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                    }
    
                    # If thresholds are provided, apply confidence filtering
                    if class_thresholds is not None:
    
                        # Check that a threshold exists for the predicted class
                        if predicted_class not in class_thresholds:
                            raise ValueError(
                                f"No confidence threshold defined for class {predicted_class}"
                            )
    
                        # Get the confidence threshold associated with the predicted class
                        threshold = class_thresholds[predicted_class]
    
                        # Accept the prediction only if its confidence is greater
                        # than or equal to the threshold of the predicted class
                        result["confidence_accepted"] = confidence >= threshold
    
                    # Add the result of this sample to the final list
                    results.append(result)
    
        return results



    """
    @brief: Evaluates the model using class-specific confidence thresholds
    
    Only predictions whose confidence is greater than or equal to the threshold
    of their predicted class are accepted.
    
    @param split: Dataset split to evaluate. It can be "train", "val" or "test"
    @param class_thresholds: Dictionary with one confidence threshold per class
    
    @return: Dictionary with filtering evaluation metrics
    """
    def evaluate_with_confidence_filter(self, split="test", class_thresholds=None):
    
        if class_thresholds is None:
            raise ValueError("class_thresholds must be provided.")
    
        # Predict all samples from the selected split and apply confidence filtering
        results = self.predict_loader(
            split=split,
            class_thresholds=class_thresholds
        )
    
        # For confidence-only filtering, the final accepted decision is the same
        # as the confidence filter decision
        for result in results:
            result["accepted"] = result["confidence_accepted"]
    
        # Total number of samples
        total_samples = len(results)
    
        # Keep only accepted predictions
        accepted_results = [
            result for result in results
            if result["accepted"]
        ]
    
        # Number of accepted and rejected predictions
        num_accepted = len(accepted_results)
        num_rejected = total_samples - num_accepted
    
        # Fraction of samples accepted by the filter
        coverage = num_accepted / total_samples if total_samples > 0 else 0.0
    
        # Accuracy on the full split before applying rejection
        original_accuracy = sum(
            result["predicted_class"] == result["true_class"]
            for result in results
        ) / total_samples if total_samples > 0 else 0.0
    
        # Accuracy only on accepted predictions
        if num_accepted > 0:
            accepted_accuracy = sum(
                result["predicted_class"] == result["true_class"]
                for result in accepted_results
            ) / num_accepted
        else:
            accepted_accuracy = 0.0
    
        return {
            "total_samples": total_samples,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "coverage": coverage,
            "original_accuracy": original_accuracy,
            "accepted_accuracy": accepted_accuracy,
            "results": results
        }


    """
    @brief: Evaluates the model using class-specific feature filtering and
            class-specific confidence thresholds
    
    A prediction is accepted only if:
        1. The sample passes the feature limits of the predicted class
        2. The confidence is greater than or equal to the threshold of the predicted class
    
    @param split: Dataset split to evaluate. It can be "train", "val" or "test"
    @param features: Dictionary with selected features for the selected split
    @param data_analysis: DataAnalysis object containing the class limits
    @param class_thresholds: Dictionary with one confidence threshold per class
    
    @return: Dictionary with filtering evaluation metrics
    """
    def evaluate_with_class_and_confidence_filter(
        self,
        split,
        features,
        data_analysis,
        class_thresholds
    ):
    
        if class_thresholds is None:
            raise ValueError("class_thresholds must be provided.")
    
        # Predict all samples and compute confidence filtering
        results = self.predict_loader(
            split=split,
            class_thresholds=class_thresholds
        )
    
        # Apply class-specific feature filtering and combine both filters
        for result in results:
    
            sample_name = result["sample_name"]
            predicted_class = result["predicted_class"]
    
            # Get selected features of this sample
            if sample_name not in features:
                raise ValueError(f"Features not found for sample: {sample_name}")
    
            sample_features = features[sample_name]
    
            # First filter: check whether the sample is compatible with
            # the limits of the predicted class
            class_filter_accepted = data_analysis.passes_class_filter(
                sample_features,
                predicted_class
            )
    
            # Second filter: check whether the confidence is high enough
            confidence_accepted = result["confidence_accepted"]
    
            # Store individual filter decisions
            result["class_filter_accepted"] = class_filter_accepted
            result["confidence_accepted"] = confidence_accepted
    
            # Final decision.
            # A prediction is accepted only if it passes both filters.
            result["accepted"] = class_filter_accepted and confidence_accepted
    
        # Total number of samples
        total_samples = len(results)
    
        # Accepted predictions after both filters
        accepted_results = [
            result for result in results
            if result["accepted"]
        ]
    
        # Rejected predictions after at least one filter
        rejected_results = [
            result for result in results
            if not result["accepted"]
        ]
    
        # Number of accepted and rejected predictions
        num_accepted = len(accepted_results)
        num_rejected = len(rejected_results)
    
        # Fraction of samples accepted by the complete system
        coverage = num_accepted / total_samples if total_samples > 0 else 0.0
    
        # Accuracy before applying rejection
        original_accuracy = sum(
            result["predicted_class"] == result["true_class"]
            for result in results
        ) / total_samples if total_samples > 0 else 0.0
    
        # Accuracy only on accepted predictions
        if num_accepted > 0:
            accepted_accuracy = sum(
                result["predicted_class"] == result["true_class"]
                for result in accepted_results
            ) / num_accepted
        else:
            accepted_accuracy = 0.0
    
        # Count how many samples each filter rejects
        rejected_by_class_filter = sum(
            not result["class_filter_accepted"]
            for result in results
        )
    
        rejected_by_confidence = sum(
            not result["confidence_accepted"]
            for result in results
        )
    
        rejected_by_both = sum(
            not result["class_filter_accepted"] and not result["confidence_accepted"]
            for result in results
        )
    
        return {
            "total_samples": total_samples,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "coverage": coverage,
            "original_accuracy": original_accuracy,
            "accepted_accuracy": accepted_accuracy,
            "rejected_by_class_filter": rejected_by_class_filter,
            "rejected_by_confidence": rejected_by_confidence,
            "rejected_by_both": rejected_by_both,
            "results": results
        }

    """
    @brief: Returns the DataLoader corresponding to the selected split
    """
    def _get_loader(self, split):

    
        loaders = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader
        }
    
        if split not in loaders:
            raise ValueError("split must be 'train', 'val' or 'test'.")
    
        loader = loaders[split]
    
        if loader is None:
            raise ValueError(f"{split} data has not been loaded.")
    
        return loader

    """
    @brief: Predicts the class of one sample and applies a confidence threshold
    depending on the predicted class

    @param sample: Dictionary with image paths for the selected channels
    @param class_thresholds: Dictionary with minimum confidence per class

    @return: predicted_class, confidence, accepted
    """
    def predict_with_filter(self, sample, class_thresholds):

    
        predicted_class, confidence = self.predict(sample)
    
        if predicted_class not in class_thresholds:
            raise ValueError(f"No confidence threshold defined for class {predicted_class}")
    
        threshold = class_thresholds[predicted_class]
        accepted = confidence >= threshold
    
        return predicted_class, confidence, accepted


    """
    Searches class-specific percentile ranges using validation data.
    
    Limits are computed from train_data.
    Performance is evaluated on validation data.
    
    @return:
        class_percentiles, best_metrics
    
        class_percentiles = None means that only confidence thresholds are used.
    """
    def tune_class_filter_percentiles(
        self,
        train_data,
        val_features,
        data_analysis,
        class_thresholds,
        percentile_candidates=None,
        min_coverage=0.75
    ):
    
        if percentile_candidates is None:
            percentile_candidates = [
                (1, 99),
                (2, 98),
                (5, 95),
                (10, 90),
                (15, 85),
            ]
    
        print("\n--------- CLASS-SPECIFIC PERCENTILE SEARCH ----------------")
    
        # First we evaluate with the confidence filter only
        best_metrics = self.evaluate_with_confidence_filter(
            split="val",
            class_thresholds=class_thresholds
        )
    
        best_score = best_metrics["accepted_accuracy"] + 0.01 * best_metrics["coverage"]
        best_class_percentiles = None
    
        print(
            f"Confidence only | "
            f"Coverage: {best_metrics['coverage']:.4f} | "
            f"Accepted acc: {best_metrics['accepted_accuracy']:.4f} | "
            f"Rejected: {best_metrics['num_rejected']}"
        )
    
        # Current best percentile configuration
        # None means no class feature filter is currently selected
        current_percentiles = None
    
        for class_idx in range(self.num_classes):
    
            print(f"\nSearching percentiles for class {class_idx}")
    
            best_candidate_for_this_class = None
            best_metrics_for_this_class = best_metrics
            best_score_for_this_class = best_score
            best_percentiles_for_this_class = current_percentiles
    
            for p_low, p_high in percentile_candidates:
    
                # If no feature filter has been selected yet, start with a neutral
                # configuration and only activate the current class.
                if current_percentiles is None:
                    candidate_percentiles = {
                        c: (0, 100)
                        for c in range(self.num_classes)
                    }
                else:
                    candidate_percentiles = current_percentiles.copy()
    
                candidate_percentiles[class_idx] = (p_low, p_high)
    
                data_analysis.compute_limits_per_class(
                    train_data,
                    class_percentiles=candidate_percentiles
                )
    
                metrics = self.evaluate_with_class_and_confidence_filter(
                    split="val",
                    features=val_features,
                    data_analysis=data_analysis,
                    class_thresholds=class_thresholds
                )
    
                coverage = metrics["coverage"]
                accepted_accuracy = metrics["accepted_accuracy"]
    
                print(
                    f"Class {class_idx} | Percentiles {p_low}-{p_high} | "
                    f"Coverage: {coverage:.4f} | "
                    f"Accepted acc: {accepted_accuracy:.4f} | "
                    f"Rejected: {metrics['num_rejected']}"
                )
    
                if coverage < min_coverage:
                    continue
    
                score = accepted_accuracy + 0.01 * coverage
    
                if score > best_score_for_this_class:
                    best_score_for_this_class = score
                    best_metrics_for_this_class = metrics
                    best_candidate_for_this_class = (p_low, p_high)
                    best_percentiles_for_this_class = candidate_percentiles.copy()
    
            if best_candidate_for_this_class is not None:
                current_percentiles = best_percentiles_for_this_class
                best_class_percentiles = current_percentiles.copy()
                best_metrics = best_metrics_for_this_class
                best_score = best_score_for_this_class
    
                print(
                    f"Selected for class {class_idx}: "
                    f"{best_candidate_for_this_class[0]}-{best_candidate_for_this_class[1]}"
                )
            else:
                print(f"No percentile filter selected for class {class_idx}")
    
        print("\nBest configuration:")
    
        if best_class_percentiles is None:
            print("Confidence only. No class percentile filter selected.")
        else:
            data_analysis.compute_limits_per_class(
                train_data,
                class_percentiles=best_class_percentiles
            )
    
            for class_idx, (p_low, p_high) in best_class_percentiles.items():
                print(f"Class {class_idx}: {p_low}-{p_high}")
    
        print(f"Coverage: {best_metrics['coverage']:.4f}")
        print(f"Accepted acc: {best_metrics['accepted_accuracy']:.4f}")
        print("-----------------------------------------------------------\n")
    
        return best_class_percentiles, best_metrics

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
        
        # Get all fields/channels of this microalga
        fields = self.data[sample_name]
    
        # Load selected microalga channels and convert them into a tensor
        x = self.load_fn(fields)
        
        # Get the class label and convert it to a PyTorch tensor
        # dtype=torch.long is required by CrossEntropyLoss
        y = torch.tensor(fields["class"], dtype=torch.long)
    
        # Return also the sample name so predictions can be linked later
        # with morphological/fluorescence features
        return x, y, sample_name
    