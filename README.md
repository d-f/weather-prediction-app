# weather-prediction-app
The NOAA has hosted hourly weather data including air temperature measurements for multiple different stations around the world. The different stations can be downloaded at: https://www.ncei.noaa.gov/data/global-hourly/access/2023/?C=S;O=D

For this demonstration application, the 99999927516.csv was used containing data from a station in Alaska.

This csv was uploaded to a local MongoDB using PyMongo and the following:

```
python upload_csv_files.py -db_name noaa_global_hourly -col_name wp_app -csv_dir /ml_projects/weather-prediction-app/noaa_data/
```
The default host name of localhost:27017 will be used, if a different one is desired, change by setting -hostname argument of upload_csv_files.py

Each row in the CSV file contains data for roughly every 5 minutes, and the temperature reading has various quality control checks.

PyMongo was used in order to filter for missing temperature values or temperature values with undesirable quality control flags:

```
python create_dataset.py -save_path /ml_projects/weather-prediction-app/raw_dataset.json
```

The above code will create a JSON file dictated by -save_path where the keys are timestamps and the values are temperature values.

These raw data are normalized and made seasonal stationary with:

```
python process_dataset.py
```


In the MySQL terminal:
```
mysql --local_infile=1 -u root -p
CREATE DATABASE weather_prediction;
USE weather_prediction;
source C:/path/to/load_combined.sql
```
In order to inspect the model performance:
```
source C:/path/to/describe_performance.sql
```

| model_save_name | batch_size | learning_rate | number_of_LSTM_layers | hidden_size | input_size | output_size | test_loss |
| --------------- | ---------- | ------------- | --------------------- | ----------- | ---------- | ----------- | --------- |
| model_5.pth.tar |         16 |      0.000001 |                     3 |         256 |         12 |          24 | 0.0273356 |


| Batch Size  | Learning Rate | LSTM Layers | Hidden Size | Test Loss | Test Loss Δ min.      |
| ----------- | ------------- | ----------- | ----------- | --------- | --------------------- |
|         16 |      0.000001 |            3 |         256 | 0.0273356 |                     0 |
|         16 |      0.000001 |            3 |         512 | 0.0298493 | 0.0025136861950159073 |
|         16 |       0.00001 |            3 |         256 | 0.0365308 |  0.009195137768983841 |
|         16 |      0.000001 |           32 |        1024 | 0.0369591 |  0.009623490273952484 |
|         16 |     0.0000001 |            3 |         256 | 0.0382041 |  0.010868437588214874 |
|         16 |      0.000001 |           10 |         256 | 0.0438808 |    0.0165451280772686 |
|         16 |      0.000001 |           10 |         512 | 0.0510757 |   0.02374005690217018 |
|         16 |          0.01 |            3 |         256 | 0.0516867 |   0.02435111254453659 |
|         16 |        0.0001 |            3 |         256 |  0.063349 |   0.03601333871483803 |
|         16 |         0.001 |            3 |         256 | 0.0647168 |   0.03738119825720787 |



To dockerize the application:
```
docker build -t weather_prediction_app .
```

In order to run the application to send POST requests:
```
docker run -p 80:80 weather_prediction_app
```





