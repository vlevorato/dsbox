# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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