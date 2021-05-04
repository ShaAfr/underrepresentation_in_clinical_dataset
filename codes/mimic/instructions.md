MIMIC-III analysis steps
=========================

### General Instruction
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
9.	You test the model using the following command. 
`python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --depth 2 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/{model_name}.state --mode test`
10.	You will get the output data (subjects, predictions, true labels) from `mimic3-benchmarks/test_predictions/{model_name}.state.csv`
11.	Take this csv file along with PATIENTS.csv and ADMISSIONS.csv (from MIMIC III) as input to `quick_analysis_from_output_csv.ipynb`. This will link samples with demographic information.
12.	The output csv file from step 10 will be taken as input to `3_results in subgroups.ipynb`. It will find balanced accuracy, accuracy, recall_C1 (recall of minority group), recall_C0 (recall of majority group), precision_C1, precision_C0, F1_Score_C1, F1_Score_C0, AUROC, PR_Curve_C1, PR_Curve_CO of 14 demographic subgroups (Whole groups, Male, Female, White, Black, Asian, Hispanic, Age<30, Age between 30 and 40, Age between 40 and 50, Age between 50 and 60, Age between 60 and 70, Age between 70 and 80, Age between 80 to 90 and Age greater than 90)

### DP Sampling (Modification of General Instructions)
1.	At step 5 of general instruction, we will change the change listfile for undersampling and oversampling. 
2.	For our proposed DP sampling in training data, use the existing training_listfile.csv file as input to the ipynb file of chosen minority group in `underrepresentation_in_clinical_dataset/codes/mimic/training_data_processing/dp/`
3.	The ipynb file will generate 14 files (2 units to 20 units added training files). 
4.	Follow the general instruction 6-11 with generated 14 training files. 
5.	Use `dp_unit_choice.ipynb` file to identify which unit has optimum repetition of under represented population. 
6.	Fing the performance of any specific underrepresented subgroup now with the chosen unit from previous step with `3_results in subgroups.ipynb` file.
