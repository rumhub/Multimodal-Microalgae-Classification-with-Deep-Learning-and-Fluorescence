from pathlib import Path
from classes.data_reader import DataReader
from classes.data_analysis import DataAnalysis
from classes.csv_writer import CSVWriter
from classes.model import Model
import classes.config as config


def main():
    
    
    # =================== CONFIGURATION ===================

    # If True, the CNN is trained from scratch and saved
    # If False, a previously saved model is loaded
    TRAIN_MODEL = False
    
    # Convert model path to a Path object
    model_path = Path(config.MODEL_PATH)
    
    if not TRAIN_MODEL:
        # Check that the saved model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}\n"
                f"Train the model first by setting TRAIN_MODEL = True."
            )
    
    
    # =================== IMAGE READING AND PREPROCESSING ===================

    # --- Read data ---
    data_reader = DataReader("../images")
    
    # Read data
    img_paths = data_reader.read_data()
    
    # --- Analyze data ---
    data_analysis = DataAnalysis()
    
    # Split global data into training, validation, and test
    train_data, val_data, test_data = data_analysis.split_train_val_test(img_paths)
    
    # Plot channels of a microalga
    data_analysis.plot_microalga_channels(train_data, 60)

    # Calculate microalgae metrics
    train_data = data_analysis.get_img_metrics(train_data)
    val_data = data_analysis.get_img_metrics(val_data)
    test_data = data_analysis.get_img_metrics(test_data)

    # Analyze only train
    data_analysis.plot_fluorescence_correlation(train_data)
    data_analysis.print_fluorescence_summary(train_data)
    data_analysis.plot_mofologic_correlation(train_data)
    data_analysis.plot_final_variables_correlation(train_data)
    
    # Filter out redundant features
    train_data = data_analysis.remove_unselected_features(train_data)
    val_data = data_analysis.remove_unselected_features(val_data)
    test_data = data_analysis.remove_unselected_features(test_data)
    
    # Compute global limits only from train
    data_analysis.compute_global_limits(train_data, p_low=2, p_high=98)

    # Global filtering
    train_data = data_analysis.global_filtering(train_data, debug=0)
    val_data = data_analysis.global_filtering(val_data, debug=0)
    test_data = data_analysis.global_filtering(test_data, debug=0)

    # Compute limits per class only from train
    data_analysis.compute_limits_per_class(train_data, p_low=5, p_high=95)

    # --- Export to csv ---
    csv_writer = CSVWriter()
    csv_writer.export_img_info_to_csv(train_data, "../data_info/train.csv")
    csv_writer.export_img_info_to_csv(val_data, "../data_info/val.csv")
    csv_writer.export_img_info_to_csv(test_data, "../data_info/test.csv")

    # Split data in images and features
    train_images, train_features = data_analysis.split_data_from_features(train_data)
    val_images, val_features = data_analysis.split_data_from_features(val_data)
    test_images, test_features = data_analysis.split_data_from_features(test_data)
    
    
    # =================== MODEL ===================
    # --- Model ---
    model = Model(
        # selected_channels= config.IMG_SUFFIXES,
        selected_channels=["amp", "phase", "flr_2"],
        num_classes= len(config.CLASS_PREFIXES),
    )
    
    model.read_data(
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        batch_size=32,
        num_workers=2
    )
    
    if TRAIN_MODEL:
        # Train the model from scratch
        model.train(num_epochs=200, learning_rate=1e-4)
        
        # Save the trained model weights
        model.save(config.MODEL_PATH)
        
        print(f"Model saved in: {config.MODEL_PATH}")
    else:
        # Load previously trained model weights.
        model.load(config.MODEL_PATH)
    
        print(f"Model loaded from: {config.MODEL_PATH}")
    
    
    test_loss, test_acc = model.evaluate(split="test")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    
    # =================== CONFIDENCE THRESHOLD FILTERING ===================

    # Compute one confidence threshold for each predicted class using validation data
    class_thresholds = model.compute_class_confidence_thresholds(
        split="val",
        min_accepted_ratio=0.8
    )
    
    print("Class confidence thresholds:")
    for class_idx, threshold in class_thresholds.items():
        print(f"Class {class_idx}: {threshold:.2f}")
    
    # Evaluate the test set using the confidence thresholds
    filtered_metrics = model.evaluate_with_confidence_filter(
        split="test",
        class_thresholds=class_thresholds
    )
    
    print(f"Total samples: {filtered_metrics['total_samples']}")
    print(f"Accepted predictions: {filtered_metrics['num_accepted']}")
    print(f"Rejected predictions: {filtered_metrics['num_rejected']}")
    print(f"Coverage: {filtered_metrics['coverage']:.4f}")
    print(f"Original accuracy: {filtered_metrics['original_accuracy']:.4f}")
    print(f"Accuracy on accepted predictions: {filtered_metrics['accepted_accuracy']:.4f}")
    
    
    
if __name__ == "__main__":
    main()

