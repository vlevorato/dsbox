# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4] - 2020-06-07

### Added

- KerasApplicationFactory generating known NN architectures for image labelling (for instance Xception)
- Add inverse_transform method for TagEncoder 
- Documentation for OPS part (Operators, Data Units)
- DataGlobalInputUnit and DataGlobalOutputUnit able to handle all dataframes APIs (Pandas, Vaex, Dask and co.)
- Tutorial notebook "Classic Auto-encoders architectures"
- Tutorial notebook "LSTM and CNN for text classification - Sentiment analysis applied to tweets in French"

### Changed

- Update keras imports to tensorflow.keras
- Add OutlierMixin inheritance to outliers class
- Add a window parameter to MADOutliers estimator

### Fixed

- Add __str__ method to multiple data units (cause crash in Apache Airflow if consulting task instance details)
- Update joblib import from sklearn.externals.joblib to joblib
- Re-aligned some pacakges importation to avoid future deprecation

## [1.3.1] - 2020-05-10

### Added

- Data Operator class to handle data operation in Apache Airflow
- Data Units which are used with Data Operators to manage data input and output streams
- DBConnector to connect RDBMS database
- PlasmaConnector for connection to Arrow Plasma store
- Calibration model techniques
- Some ensemble models (Rotation Forest)
- Model explanation methods (feature contribution and prediction interval in Random Forest)
- Categorical and time series pre-built feature engineering
- Feature selection methods (greedy)
- Simple markov chains, supervised and unsupervised Hidden Markov Models
- End-to-end NN architectures
- Outlier detections techniques (MAD, FFT, Gaussian Process, ...)
- NLP models
- Pre-built function to display common graphics (Confusion matrix, ROC curve, prediction interval viz)
- Metrics score
- Miscellaneous functions (object persistence,  file name generator, ...)