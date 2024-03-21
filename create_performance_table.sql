CREATE TABLE Models(
    model_save_name VARCHAR(255) PRIMARY KEY,
    batch_size INT,
    learning_rate FLOAT,
    number_of_LSTM_layers INT,
    hidden_size INT,
    input_size INT,
    output_size INT, 
    test_loss FLOAT
);