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




mysql --local_infile=1 -u root -p

The results from the SQL query suggest that a larger hidden sizes and additional learning rates around 1e-5 might be useful to train.
| Batch Size  | Learning Rate | LSTM Layers | Hidden Size | Test Loss             |
| ----------- | ------------- | ----------- | ----------- | --------------------- |
| 16          |  1e-5         | 5           | 1024        | 1.6 x 10<sup>-5</sup> |
| 16          |  1e-5         | 5           | 512         | 2.6 x 10<sup>-5</sup> |
| 16          |  5e-5         | 5           | 512         | 8.7 x 10<sup>-5</sup> |




| LSTM Layers | Test MSE              | difference from minimum |
| ----------- | --------------------- | ----------------------- | 
| 5           | 1.6 x 10<sup>-5</sup> | 0                       |  
| 5           | 2.6 x 10<sup>-5</sup> | 9.6 x 10<sup>-6</sup>   | 
| 5           | 8.7 x 10<sup>-5</sup> | 7.0 x 10<sup>-5</sup>   |
| 5           | 0.03                  | 0.04                    |  
| 5           | 0.04                  | 0.04                    | 
| 5           | 0.05                  | 0.05                    | 
| 5           | 0.05                  | 0.05                    |  
| 5           | 0.11                  | 0.11                    | 
| 10          | 0.11                  | 0.11                    | 
| 5           | 0.13                  | 0.13                    |




```
docker build -t weather_prediction_app .
```


```
docker run -p 80:80 weather_prediction_app
```





