SEER instructions
=========================

1. Request access to SEER data on their website (https://seer.cancer.gov/data/access.html). Our experiments used 1973-2014 (November 2016 Submission). The ASCII data file is available for download upon request. Newer submission of data is also available on SEER website (https://seer.cancer.gov/data-software/documentation/seerstat/). 


2. Install all of the dependencies. The original experiment requires all dependencies in the requirements.txt to be installed. We use Python 3.9 and TensorFlow 2 (instead of TensorFlow 1 in the original version of code, due to compatibility issues). Some of the sampling methods we tried require the imbalanced-learn library (imported as imblearn). 


3. Run the experiment. To run the original experiment, use a command such as 

`python3 main.py --incidences breast/BREAST.txt --specifications example/read.seer.research.nov2016.sas --cases breast/breast_all_cases.csv --task survival60 --oneHotEncoding --model MLP --mlpLayers 2 --mlpWidth 20 --mlpEpochs 1 --mlpDropout 0.1 --importance --plotData --plotResults`



4. Sampling and DP. Use the --sampling parameter to choose a sampling method. The default choice is no sampling. To run DP, use --sampling DP and specify the target subgroup and DP unit. An example command for running a DP experiment:

`python3 main.py --incidences breast/BREAST.TXT --specifications example/read.seer.research.nov2016.sas --cases breast/breast_all_cases.csv --task survival60 --oneHotEncoding --model MLP --mlpLayers 2 --mlpWidth 20 --mlpEpochs 25 --mlpDropout 0.1 --sampling DP --DPUnits 8 --subgroup Asian --importance --test`

Full list of available options can be found by running

`python3 main.py -h`


5. Generate results. Validation results and test results for each epoch will be generated after the experiment. A checkpoint of epoch will also be saved. The code in result analysis folder can be used to do the calibration and threshold tuning as described in the paper. DP has a separate script for model selection because the procedure is slightly different from others.


