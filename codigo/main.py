from classes.data_reader import DataReader
from classes.data_analysis import DataAnalysis
from classes.csv_writer import CSVWriter

def main():
    # --- Read data ---
    data_reader = DataReader("../datos_cultivos")
    
    # Read data
    img_paths = data_reader.read_data()
    
    # Remove incomplete images (missing channels)
    img_paths = data_reader.clean_incomplete_data(img_paths)
    
    # --- Analyze data ---
    data_analysis = DataAnalysis()
    img_paths = data_analysis.get_img_metrics(img_paths)
    
    # Split global data into training, validation, and test
    train_data, val_data, test_data = data_analysis.split_train_val_test(img_paths)
    
    # Clean training data
    train_data = data_analysis.clean_data(train_data, debug=1)
    

    # --- Export to cv ---
    csv_writer = CSVWriter()
    csv_writer.export_img_info_to_csv(img_paths, "../data_info/out.csv")
    
    
if __name__ == "__main__":
    main()

