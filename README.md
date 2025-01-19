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

Each row in the CSV file contains data for roughly every 5 minutes, and the temperature reading has various quality control checks.

PyMongo was used in order to filter for missing temperature values or temperature values with undesirable quality control flags:

```
python create_datasets.py -host_name "localhost:27017/" -db_name "noaa_global_hourly" -col_name "wp_app" -save_path "./raw_dataset.json"
```

The above code will create a JSON file dictated by -save_path where the keys are timestamps and the values are temperature values.

These raw data are normalized, windowed, partitioned and made seasonally stable with:

```
python process_dataset.py -raw_data_filepath "./raw_dataset.json" -n_save_filepath "./normalized_dataset.json" -val_prop 0.1 -input_width 12 -output_width 24 -data_prep_json "./data_prep.json"
```

![seasonality](https://github.com/d-f/weather-prediction-app/assets/118086192/ad833d1e-23da-43ca-9481-a253fc84e4a9)

Figure 1: Seasonality decomposition results 


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
SET GLOBAL local_infile=ON;
CREATE DATABASE weather_prediction;
USE weather_prediction;
source C:/path/to/load_combined.sql
```

```
source C:/path/to/describe_performance.sql
```
Output:

| batch_size | learning_rate | number_of_LSTM_layers | hidden_size | Mean Squared Error  | MSE difference_from_min   |
|----------- | ------------- | --------------------- | ----------- | ---------- | --------------------- |
|          8 |        0.0008 |                     4 |        1024 | 0.00298405 |                     0 |
|          8 |        0.0007 |                    32 |         512 | 0.00465322 | 0.0016691710334271193 |
|         16 |       0.00001 |                    32 |         512 | 0.00529043 |  0.002306379145011306 |
|         16 |         0.001 |                    64 |        1024 |  0.0055018 | 0.0025177544448524714 |
|         16 |         0.001 |                    32 |        1024 | 0.00568289 |  0.002698844065889716 |
|         32 |         0.001 |                    32 |         512 | 0.00594506 |  0.002961012301966548 |
|         16 |         0.001 |                    32 |         512 |  0.0069524 |  0.003968354547396302 |
|         16 |        0.0005 |                    32 |         256 | 0.00760935 |  0.004625295987352729 |
|          8 |        0.0001 |                     8 |         128 | 0.00999573 |  0.007011685287579894 |
|         16 |        0.0001 |                    32 |         512 |  0.0151662 |   0.01218219450674951 |

Table 1: Results from the first query in describe_performance.sql

| number of LSTM layers | Mean Squared error  | MSE difference from min   |
| --------------------- | ---------- | --------------------- |
|                     4 | 0.00298405 |                     0 |
|                    32 | 0.00465322 | 0.0016691710334271193 |
|                    32 | 0.00529043 |  0.002306379145011306 |
|                    64 |  0.0055018 | 0.0025177544448524714 |
|                    32 | 0.00568289 |  0.002698844065889716 |
|                    32 | 0.00594506 |  0.002961012301966548 |
|                    32 |  0.0069524 |  0.003968354547396302 |
|                    32 | 0.00760935 |  0.004625295987352729 |
|                     8 | 0.00999573 |  0.007011685287579894 |
|                    32 |  0.0151662 |   0.01218219450674951 |

Table 2: Results from the second query in describe_performance.sql

| model_save_name         | batch_size | learning_rate | number_of_LSTM_layers | hidden_size | input_size | output_size | MSE  |
| ----------------------- | ---------- | ------------- | --------------------- | ----------- | ---------- | ----------- |------------|
| stable_model_10.pth.tar |          8 |        0.0008 |                     4 |        1024 |         12 |          24 | 0.00298405 |

Table 3: Results from the third query in describe_performance.sql

To dockerize the application:
```
docker build -t weather_prediction_app .
```

In order to run the application to send POST requests (port 80 used for this example):
```
docker run -p 80:80 weather_prediction_app
```
