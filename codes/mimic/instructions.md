MIMIC-III analysis steps
=========================

### General Instruction (in-hopsital mortality prediction)
1.	MIMIC III dataset can be downloaded from https://mimic.physionet.org/
2.	We construct benchmark machine learning dataset from https://github.com/YerevaNN/mimic3-benchmarks using MIMIC III dataset
3.	In the Benchmark GitHub, please follow the six steps specified in “Building a benchmark” subsection to create benchmark machine learning dataset for In-hospital mortality and decompensation prediction task. 
4.	In the mimic3-benchmarks folder, put the ‘run_command.sh’ file.  (You can find it from: https://github.com/ShaAfr/underrepresentation_in_clinical_dataset/blob/main/codes/mimic/result_processing/run_command.sh)
5.	The training data is in mimic3-benchmarks/data/in-hospital-mortality/train_listfile.csv
6.	Run the following command to save the output in run_output.txt file.
 `sh run_command.sh &>> run_output.txt`
7.	Determine the epoch from where we get the final model. Input run_output.txt in 1_epoch_to_consider.ipynb (`underrepresentation_in_clinical_dataset/codes/mimic/result_processing/1_epoch_to_consider.ipynb`)
8.	Check the `run_output.txt` file and find out the model name for the specific epoch. For example, is the chosen epoch is 60, you can see a similar line in the run_output.txt file.
`Epoch 00060: saving model to mimic3models/in_hospital_mortality/keras_states/{model_name}.state`
9.	You test the specific model using the following command. 
`python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --depth 2 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/{model_name}.state --mode test`
10.	You will get the output data (subjects, predictions, true labels) from `mimic3-benchmarks/test_predictions/{model_name}.state.csv`
11.	Repeat step 9 and 10 for getting the performance of validation set as well.
12.	Put the validation csv file and test csv file in `csv_files/val/` and `csv_files/test/` folder.
13.	We calibrate the predictions and choose optimal threshold result using `2_calibrated_sampling_and_optimal_threshold.ipynb`
14.	Use PATIENTS.csv and ADMISSIONS.csv (from MIMIC III) as input to `3_quick_analysis.ipynb`. This will link samples with demographic information.
15.	For DP model, `3_dp_choose_unit (only for DP -- run before 3_automated analysis_on_table).ipynb` will then choose dynamically which DP unit to choose. For other model, skip step 15.
16.	The output csv file from previous steps will be taken as input to `4_automated_analysis_on_table_ihm`. It will find different metrices, i.e., balanced accuracy, accuracy, recall_C1 (recall of minority group), recall_C0 (recall of majority group), precision_C1, precision_C0, F1_Score_C1, F1_Score_C0, AUROC, PR_Curve_C1, PR_Curve_CO, MCC, FPR, FNR of 14 demographic subgroups (Whole groups, Male, Female, White, Black, Asian, Hispanic, Age<30, Age between 30 and 40, Age between 40 and 50, Age between 50 and 60, Age between 60 and 70, Age between 70 and 80, Age between 80 to 90 and Age greater than 90)
17.	Compute different graph results using `5_ResultExtract*.csv` files
