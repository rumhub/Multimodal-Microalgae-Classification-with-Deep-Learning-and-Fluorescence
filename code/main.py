import random
import numpy as np
import torch
from pathlib import Path
from classes.data_reader import DataReader
from classes.data_analysis import DataAnalysis
from classes.csv_writer import CSVWriter
from classes.model import Model
import classes.config as config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using seed: {seed}")

def main():
    
    
    # =================== CONFIGURATION ===================

    SEED = 42
    set_seed(SEED)

    # If True, the CNN is trained from scratch and saved
    # If False, a previously saved model is loaded
    TRAIN_MODEL = True
    
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
    img_paths = data_reader.read_data()
    
    # --- Analyze data ---
    data_analysis = DataAnalysis()
    
    # Rebalance classes
    # img_paths = data_analysis.balance_classes_to_min_count(img_paths)

    # Split global data into training, validation, and test
    train_data, val_data, test_data = data_analysis.split_train_val_test(img_paths)
    
    # Plot channels of a microalga
    data_analysis.plot_microalga_channels(train_data, 60)

    # Calculate microalgae metrics
    train_data = data_analysis.get_img_metrics(train_data)
    val_data = data_analysis.get_img_metrics(val_data)
    test_data = data_analysis.get_img_metrics(test_data)

    # Analyze fluorescence channels composition (only train)
    data_analysis.verify_flu_composition(train_data)    # Verify that FLU = RED_CHANNEL(FLU_1 + FLU2 + FLU3)
    data_analysis.analyze_fluorescence_rgb_channels(train_data)     # See which rgb channel has more info in (FLU_1, FLU2, FLU3)
    
    # Filter out redundant fluorescence channels (flu)
    train_data = data_analysis.remove_unselected_image_channels(train_data)
    val_data = data_analysis.remove_unselected_image_channels(val_data)
    test_data = data_analysis.remove_unselected_image_channels(test_data)
    
    # Analize correlation (only train)
    data_analysis.plot_fluorescence_correlation(train_data)
    fluorescence_summary = data_analysis.print_fluorescence_summary(train_data)
    data_analysis.plot_mofologic_correlation(train_data)
    data_analysis.plot_final_variables_correlation(train_data)

    # How non-selected features
    data_analysis.show_unselected_features_histograms(
        train_data,
        save_dir="../data_info/plots/unselected_features"
    )    

    data_analysis.show_all_features_distributions_by_class(
        train_data,
        save_dir="../data_info/plots/features_by_class"
    )

    # Filter out redundant features
    train_data = data_analysis.remove_unselected_features(train_data)
    val_data = data_analysis.remove_unselected_features(val_data)
    test_data = data_analysis.remove_unselected_features(test_data)
    
    # Compute global limits only from train
    data_analysis.compute_global_limits(train_data, p_low=1, p_high=99)

    # Global filtering
    train_data = data_analysis.global_filtering(train_data, debug=1, save_dir="../data_info/plots/global_filtering")
    val_data = data_analysis.global_filtering(val_data, debug=0)
    test_data = data_analysis.global_filtering(test_data, debug=0)

    # =================== EXPORT TO CSV ===================
    print("--------- CSV WRITER ----------------")
    csv_writer = CSVWriter()
    csv_writer.export_img_info_to_csv(train_data, "../data_info/train.csv")
    csv_writer.export_img_info_to_csv(val_data, "../data_info/val.csv")
    csv_writer.export_img_info_to_csv(test_data, "../data_info/test.csv")
    csv_writer.export_summary_to_csv(fluorescence_summary, "../data_info/fluorescence_summary.csv")

    # Split data in images and features
    train_images, train_features = data_analysis.split_data_from_features(train_data)
    val_images, val_features = data_analysis.split_data_from_features(val_data)
    test_images, test_features = data_analysis.split_data_from_features(test_data)
    
    
    # =================== MODEL ===================
    # --- Model ---
    model = Model(
        # selected_channels= config.IMG_SUFFIXES,
        # selected_channels=["amp", "phase", "flr_2"],
        selected_channels=config.SELECTED_IMG_SUFFIXES,
        # selected_channels=["amp", "flr_1", "flr_2", "mask", "phase"],
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
        model.train(num_epochs=300, learning_rate=1e-4)
        
        # Plot training curves
        model.plot_training_curves(save_dir="../data_info/plots/training")
        
        # Save the trained model weights
        model.save(config.MODEL_PATH)
        
        print(f"Model saved in: {config.MODEL_PATH}")
    else:
        # Load previously trained model weights.
        model.load(config.MODEL_PATH)
    
        print(f"Model loaded from: {config.MODEL_PATH}")
    
    train_loss, train_acc = model.evaluate(split="train")
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    val_loss, val_acc = model.evaluate(split="val")
    print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

    test_loss, test_acc = model.evaluate(split="test")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    
    # =================== CLASS FILTER + CONFIDENCE THRESHOLD FILTERING ===================
    
    # Compute one confidence threshold for each predicted class using validation data
    class_thresholds = model.compute_class_confidence_thresholds(
        split="val", min_accepted_ratio=0.8, debug=True
    )
    print("Class confidence thresholds:")
    for class_idx, threshold in class_thresholds.items():
        print(f"Class {class_idx}: {threshold:.2f}")
    
    # Compute and select best percentile filter per class
    class_percentiles, val_filter_metrics = model.tune_class_filter_percentiles(
        train_data=train_data,
        val_features=val_features,
        data_analysis=data_analysis,
        class_thresholds=class_thresholds,
        percentile_candidates=[
            (1, 99),
            (2, 98),
            (5, 95),
            (10, 90),
            (15, 85),
        ],
        min_coverage=0.80
    )
    
    if class_percentiles is None:
        print("Selected filter: confidence thresholds only")
    else:
        print("Selected class filter percentiles:")
        for class_idx, percentiles in class_percentiles.items():
            p_low, p_high = percentiles
            print(f"Class {class_idx}: {p_low}-{p_high}")
    
    print(f"Validation coverage: {val_filter_metrics['coverage']:.4f}")
    print(f"Validation accepted accuracy: {val_filter_metrics['accepted_accuracy']:.4f}")
    

    # =================== FINAL MODEL PERFORMANCE ===================
    print("------------------ FINAL RESULTS ------------------")
    
    if class_percentiles is None:
        filtered_metrics = model.evaluate_with_confidence_filter(
            split="test",
            class_thresholds=class_thresholds
        )
    
        print("\nTest results with confidence filter only:")
    
    else:
        filtered_metrics = model.evaluate_with_class_and_confidence_filter(
            split="test",
            features=test_features,
            data_analysis=data_analysis,
            class_thresholds=class_thresholds
        )
    
        print("\nTest results with class filter + confidence filter:")
    
    print(f"Total samples: {filtered_metrics['total_samples']}")
    print(f"Accepted predictions: {filtered_metrics['num_accepted']}")
    print(f"Rejected predictions: {filtered_metrics['num_rejected']}")
    print(f"Coverage: {filtered_metrics['coverage']:.4f}")
    print(f"Original accuracy: {filtered_metrics['original_accuracy']:.4f}")
    print(f"Accuracy on accepted predictions: {filtered_metrics['accepted_accuracy']:.4f}")
    
    if class_percentiles is not None:
        print(f"Rejected by class filter: {filtered_metrics['rejected_by_class_filter']}")
        print(f"Rejected by confidence: {filtered_metrics['rejected_by_confidence']}")
        print(f"Rejected by both: {filtered_metrics['rejected_by_both']}")
    
    
    
if __name__ == "__main__":
    main()

