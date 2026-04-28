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
    
    data_analysis.plot_fluorescence_correlation(train_data)
    data_analysis.print_fluorescence_summary(train_data)

    
    # Calculate percentiles
    data_analysis.compute_limits_per_class(train_data, p_low=10, p_high=90)
    data_analysis.compute_global_limits(train_data, p_low=5, p_high=95)

    # Clean training data
    train_data = data_analysis.global_filtering(train_data, debug=1)
    

    # --- Export to cv ---
    csv_writer = CSVWriter()
    csv_writer.export_img_info_to_csv(img_paths, "../data_info/out.csv")
    
    
if __name__ == "__main__":
    main()

