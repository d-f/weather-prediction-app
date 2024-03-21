# weather-prediction-app
The NOAA has hosted hourly weather data including air temperature measurements for multiple different stations around the world. The different stations can be downloaded at: https://www.ncei.noaa.gov/data/global-hourly/access/2023/?C=S;O=D

For this demonstration application, the 99999927516.csv was used containing data from a station in Alaska.

```
git clone https://github.com/d-f/weather-prediction-app.git C:\\ml_projects\\weather_prediction_app\\code\\
cd C:\\ml_projects\\weather_prediction_app\\code\\weather-prediction-app\\
```

This csv was uploaded to a local MongoDB using PyMongo and the following:

```
python upload_csv.py -host_name "localhost:27017/" -db_name noaa_global_hourly -col_name wp_app -csv_dir "C:\\personal_ML\\weather_prediction_app\\code\\noaa_data\\"
```
The default host name of localhost:27017 will be used, if a different one is desired, change by setting -hostname argument of create_datasets.py

Each row in the CSV file contains data for roughly every 5 minutes, and the temperature reading has various quality control checks.

PyMongo was used in order to filter for missing temperature values or temperature values with undesirable quality control flags:

```
python create_datasets.py -host_name "localhost:27017/" -db_name "noaa_global_hourly" -col_name "wp_app" -save_path "./raw_dataset.json"
```

The above code will create a JSON file dictated by -save_path where the keys are timestamps and the values are temperature values.

These raw data are normalized and made seasonal stationary with:

```
python process_dataset.py -raw_data_filepath "./raw_dataset.json" -n_save_filepath "./normalized_dataset.json" -val_prop 0.1 -input_width 12 -output_width 24 -data_prep_json "./data_prep.json"
```

In order to train models:
```
python train_lstm.py -output_size 24 -input_size 12 -hidden_size 1024 -dataset_json "./normalized_dataset.json" -batch_size 16 -lr 1e-6 num_layers 32 -num_epochs 256 -model_save_name model_1.pth.tar -result_dir "./results/" -patience 5
```

To describe performance of models with MySQL installed:
```
mysql --local_infile=1 -u root -p
```
Enter password
```
CREATE DATABASE weather_prediction;
USE weather_prediction;
source C:/path/to/load_combined.sql
```

```
source C:/path/to/describe_performance.sql
```
Output:
| model_save_name | batch_size | learning_rate | number_of_LSTM_layers | hidden_size | input_size | output_size | test_loss |
| --------------- | ---------- | ------------- | --------------------- | ----------- | ---------- | ----------- | --------- |
| model_5.pth.tar |         16 |      0.000001 |                     3 |         256 |         12 |          24 | 0.0273356 |

| Batch Size  | Learning Rate | LSTM Layers | Hidden Size | Test Loss | Test Loss Δ min.      |
| ----------- | ------------- | ----------- | ----------- | --------- | --------------------- |
|          16 |      0.000001 |           3 |         256 | 0.0273356 |                     0 |
|          16 |      0.000001 |           3 |         512 | 0.0298493 | 0.0025136861950159073 |
|          16 |       0.00001 |           3 |         256 | 0.0365308 |  0.009195137768983841 |
|          16 |      0.000001 |          32 |        1024 | 0.0369591 |  0.009623490273952484 |
|          16 |     0.0000001 |           3 |         256 | 0.0382041 |  0.010868437588214874 |
|          16 |      0.000001 |          10 |         256 | 0.0438808 |    0.0165451280772686 |
|          16 |      0.000001 |          10 |         512 | 0.0510757 |   0.02374005690217018 |
|          16 |          0.01 |           3 |         256 | 0.0516867 |   0.02435111254453659 |
|          16 |        0.0001 |           3 |         256 |  0.063349 |   0.03601333871483803 |
|          16 |         0.001 |           3 |         256 | 0.0647168 |   0.03738119825720787 |

To dockerize the application:
```
docker build -t weather_prediction_app .
```

In order to run the application to send POST requests (port 80 used for this example):
```
docker run -p 80:80 weather_prediction_app
```
