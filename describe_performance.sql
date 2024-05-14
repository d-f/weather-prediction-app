-- Number of LSTM layers with hidden size with test loss to see if there are any patterns
SELECT batch_size, learning_rate, number_of_LSTM_layers, hidden_size, test_loss, (test_loss - MIN(test_loss) OVER ()) as difference_from_min
    FROM Models 
    ORDER BY test_loss ASC;

-- See how much of an effect # of lSTM layers has on test loss
SELECT number_of_LSTM_layers, test_loss, (test_loss - MIN(test_loss) OVER ()) as difference_from_min
    FROM Models
    ORDER BY difference_from_min ASC;

-- Select lowest test loss
SELECT *
    FROM Models
    WHERE test_loss IN (SELECT MIN(test_loss) FROM MODELS);
