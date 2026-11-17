There are two versions of the code: 'main' (without a filter) and 'main2' (which employs a Savitzky-Golay filter). For the article, we opted for the version without a filter, as our data analysis indicated that filtering was unnecessary.

We utilized key libraries—NumPy, Pandas, and PyTorch—for data manipulation and the development of deep learning models. Battery degradation data, stored in a CSV file, was loaded using Pandas. The dataset includes critical features such as Capacity, Internal Resistance, Charging Time, and State of Health (SoH)—all of which serve as primary indicators for assessing battery performance and degradation. The data was then normalized using MinMaxScaler; normalization is essential for accelerating the convergence of neural network models and preventing any single feature from dominating the others. Subsequently, the data was converted into PyTorch tensors for use in model training. We then structured the data into sequences of a specific length, preparing it for use in sequential models such as RNNs, LSTMs, MLPs, and GRUs.

Once prepared, the data was split into training and testing sets using an 80:20 ratio. This ensures that the model is evaluated on previously unseen data. The dataset was then wrapped in a `TensorDataset` and passed into a `DataLoader`, which handles batching and data shuffling—steps that help accelerate training and mitigate the risk of overfitting.

Four distinct neural network architectures were defined for comparison:
1. MLP (Multi-Layer Perceptron): A feedforward neural network with fully connected layers and no memory, serving as the baseline model.
2. RNN (Recurrent Neural Network): Incorporates memory through recurrent connections, enabling the recognition of patterns in sequential data; however, RNNs are susceptible to the vanishing gradient problem.
3. LSTM (Long Short-Term Memory): An enhanced version of the RNN that utilizes a gating mechanism, allowing it to retain long-term information. 4. GRU (Gated Recurrent Unit) is similar to LSTM but computationally simpler, often delivering comparable performance.
All models utilize a linear output layer to generate final predictions.

A single training function is used for all models. This function employs Mean Squared Error (MSE) as the loss function and the Adam optimizer to update model weights. Training is conducted over several epochs, with the loss recorded at each epoch, allowing for the assessment of model convergence. Using a consistent training loop ensures that the evaluation across different models is fair and uniform.

Each model (MLP, RNN, LSTM, GRU) is initialized and trained using the aforementioned training function. The training process spans 50 epochs, providing each model the opportunity to learn battery degradation patterns from the time-series data. By examining the training results, we can determine which models converge more rapidly and demonstrate greater learning effectiveness.

Following training, each model is evaluated using test data. The predicted values ​​and actual values ​​are rescaled to their original range by applying the inverse transform of the MinMaxScaler. Evaluation is performed using several metrics:
1. MAE (Mean Absolute Error) measures the average magnitude of errors.
2. RMSE (Root Mean Squared Error) assigns a higher penalty to larger errors.
3. R-squared (R²) measures how well the model explains the variance in the target data.
4. Relative Error (RE) indicates the magnitude of the error relative to the actual value.

Finally, the evaluation results for all models are displayed, facilitating a direct comparison. This helps determine which model is the most accurate in capturing battery degradation patterns and effectively predicting the Remaining Useful Life (RUL). Theoretically, RNN or LSTM models are expected to perform better due to their capacity to capture long-term dependencies within sequential data.
