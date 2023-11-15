https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/discussion

https://www.youtube.com/watch?v=NEaUSP4YerM

https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

https://towardsdatascience.com/anomaly-fraud-detection-a-quick-overview-28641ec49ec1

https://towardsdatascience.com/boxplot-for-anomaly-detection-9eac783382fd


Machine learning algorithms used for anomaly detection can be broadly categorized into several types, each with its strengths and weaknesses. The choice of algorithm often depends on the characteristics of the data and the specific requirements of the application. Here are some common types of machine learning algorithms used for anomaly detection:

1. **Statistical Methods:**
   - **Z-Score or Standard Score:** Measures how many standard deviations a data point is from the mean.
   - **Quartile Range (IQR):** Identifies outliers based on the interquartile range.

2. **Distance-Based Methods:**
   - **K-Nearest Neighbors (KNN):** Determines anomalies based on the distance to their k-nearest neighbors.
   - **Isolation Forest:** Constructs isolation trees to isolate anomalies efficiently.

3. **Density-Based Methods:**
   - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Identifies anomalies as points in low-density regions.
   - **LOF (Local Outlier Factor):** Measures the local density deviation of a data point with respect to its neighbors.

4. **Clustering Methods:**
   - **K-Means Clustering:** Identifies anomalies as data points that do not belong to any cluster.
   - **DBSCAN:** Can be used for clustering and identifying anomalies simultaneously.

5. **Supervised Learning:**
   - **One-Class SVM (Support Vector Machine):** Trains on the "normal" class and detects anomalies as deviations from this norm.
   - **Neural Networks:** Autoencoders can be used for unsupervised anomaly detection by reconstructing input data and identifying significant reconstruction errors.

6. **Ensemble Methods:**
   - **Random Forest:** Can be adapted for anomaly detection by identifying outliers in the decision forest.

7. **Time Series Methods:**
   - **ARIMA (AutoRegressive Integrated Moving Average):** Used for time series data, detecting anomalies based on deviations from predicted values.
   - **Prophet:** Developed by Facebook for forecasting, Prophet can be adapted to identify unusual patterns in time series data.

8. **Deep Learning:**
   - **Variational Autoencoders (VAE):** A type of autoencoder that can learn the distribution of normal data and identify anomalies.
   - **Long Short-Term Memory (LSTM) Networks:** Particularly effective for time series data, LSTMs can capture long-term dependencies and identify anomalies in sequential patterns.

It's important to note that the choice of algorithm depends on factors such as the nature of the data, the presence of labeled data, the dimensionality of the problem, and the specific characteristics of anomalies you are trying to detect. It's often beneficial to experiment with multiple algorithms and fine-tune their parameters to achieve the best results for a given application.