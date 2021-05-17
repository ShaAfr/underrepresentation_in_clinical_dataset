SEER instructions
=========================

1. Request access to SEER data on their website (https://seer.cancer.gov/data/access.html). Our experiments used 1973-2014 (November 2016 Submission). The ASCII data file is available for download upon request. Newer submission of data is also available on SEER website (https://seer.cancer.gov/data-software/documentation/seerstat/). The newest submission is 1975-2018 (November 2020 Submission).


2. Install all of the dependencies. The original experiment requires all dependencies in the requirements.txt to be installed. Some of the sampling methods we tried require the imbalanced-learn library (imported as imblearn). We use Python 3.6 for the experiments. A higher version may have incompatibility with the TensorFlow version used in provided code. If there is a NumPy incompatibility after installing imblearn, please try to reinstall the specified version in requirements.txt.


3. Run the experiment. To run the original experiment, use a command such as
`python3 main.py --incidences breast/BREAST.txt --specifications example/read.seer.research.nov2016.sas --cases breast/breast_all_cases.csv --task survival60 --oneHotEncoding --model MLP --mlpLayers 2 --mlpWidth 20 --mlpEpochs 1 --mlpDropout 0.1 --importance --plotData --plotResults`


4. Sampling. To run an experiment with sampling, please uncomment the corresponding code in the experiment.py file. For example, to run an experiment with SMOTE, uncomment
self.train_x, self.train_y = SMOTE().fit_resample(self.train_x, self.train_y)
and then use the same command.


5. DP. To run DP experiments, first, uncomment
`np.save(self.output_directory + '/X_train.npy', self.train_x)`
`np.save(self.output_directory + '/y_train.npy', self.train_y)`
in experiment.py to save a copy of the training data. (Comment them out again after having the data.) Next, use minority_selection.ipynb to generate minority data needed. Then uncomment corresponding code in experiment.py to add desirable amount of chosen minority data. For example,
`self.train_x, self.train_y = duplicate_asian_minority(3, self.train_x, self.train_y)`
duplicates Asian minority data to 3 times as much as the original data. Note this is 2 additional units. All duplicate functions do the same thing with different minority data. Use or modify as needed.


6. Generate results. After running an experiment, a file named test_results_data_frame.csv will be generated, containing the prediction results for testing data. Then use the code in test_result.ipynb to generate the results using different metrics for different subgroups. Please note that originally Class 1 for breast cancer prediction is the survival class. We flipped it to Class 0 in the article to be consistent with other tasks. The results generated here still reflect the original setting.

