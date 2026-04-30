from classes.data_reader import DataReader
from classes.data_analysis import DataAnalysis
from classes.csv_writer import CSVWriter

def main():
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

    # --- Export to cv ---
    csv_writer = CSVWriter()
    csv_writer.export_img_info_to_csv(train_data, "../data_info/train.csv")
    csv_writer.export_img_info_to_csv(val_data, "../data_info/val.csv")
    csv_writer.export_img_info_to_csv(test_data, "../data_info/test.csv")

    # Split data in images and features
    train_images, train_features = data_analysis.split_data_from_features(train_data)
    val_images, val_features = data_analysis.split_data_from_features(val_data)
    test_images, test_features = data_analysis.split_data_from_features(test_data)
    
    
if __name__ == "__main__":
    main()

