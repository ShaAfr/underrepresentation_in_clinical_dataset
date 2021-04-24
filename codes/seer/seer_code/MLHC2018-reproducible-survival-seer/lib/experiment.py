import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve, auc, \
    mean_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from lib.sampling import distant_method, replicated_oversampling, duplicate_black, duplicate_30_40,\
    duplicate_black_minority, duplicate_30_40_minority, gamma_oversampling, duplicate_asian_minority, \
    duplicate_40_50_minority, combined_DP, duplicate_white_minority, duplicate_hispanic_minority, \
    duplicate_age_minority, duplicate_gender_minority, cross_group_DP, remove_race_features
from sklearn.utils import shuffle

mpl.use('Agg')
import matplotlib.pyplot as plt


class Experiment:
    """ Class for main functionality. """

    def __init__(self, data, model, task, valid_ratio, test_ratio, model_type, encodings, encode_categorical_inputs,
                 plot_results, output_directory):
        """ Initialize main functionality and split data according to given ratios. """
        self.model = model
        self.model_type = model_type
        self.task = task

        self.plot_results = plot_results
        self.output_directory = output_directory

        input_columns = list(data.frame)
        
        ################################
        # write cols to a file
        """
        with open(self.output_directory + 'input_columns.txt', 'w') as f:
            for i in input_columns:
                f.write(i + '\n')
        """
        ################################
        
        x, y = [], []
        if task in ['mort12', 'mort60']:
            n = int(task[-2:])
            input_columns.remove("Survival months")
            # inputs
            x = data.frame[input_columns].as_matrix().astype(np.float32)
            # labels
            y = data.frame["Survival months"].as_matrix().reshape((data.frame["Survival months"].as_matrix().shape[0],))
            # Survival month must be scaled explicitly by max_survival_month.
            # If scaling it with the max value and the max is smaller than 12 or 60, this leads to wrong labels.
            y = y / n
        elif task in ['survival12', 'survival60']:
            n = int(task[-2:])
            input_columns.remove("Survived cancer for " + str(n) + " months")
            
            
            # data.frame is the original dataset
            
            class_1 = data.frame[data.frame["Survived cancer for " + str(n) + " months"] == 1]
            count_class_1 = class_1.shape[0]
            class_0 = data.frame[data.frame["Survived cancer for " + str(n) + " months"] == 0]
            count_class_0 = class_0.shape[0]
            #print('count_class_1', count_class_1)
            #print('count_class_0', count_class_0)   
            
            
            # inputs
            x = data.frame[input_columns].as_matrix().astype(np.float32)
            # labels
            y = data.frame["Survived cancer for " + str(n) + " months"].as_matrix().reshape(
                (data.frame["Survived cancer for " + str(n) + " months"].as_matrix().shape[0],)).astype(np.int32)
                
            
        self.total_1 = np.count_nonzero(y)
        self.total_0 = y.shape[0] - self.total_1

        # Fix random state to obtain same sets across experiments
        self.train_x, self.test_x, self.train_y, self.test_y \
            = train_test_split(x, y, test_size=valid_ratio + test_ratio, shuffle=True, random_state=73)

        # Fix random state to obtain same sets across experiments
        self.valid_x, self.test_x, self.valid_y, self.test_y \
            = train_test_split(self.test_x, self.test_y, test_size=(test_ratio / (valid_ratio + test_ratio)),
                               shuffle=True, random_state=63)
                               
                               
        #############################
        # save train / test files for additional experiments
        #np.save(self.output_directory + '/X_train.npy', self.train_x)
        #np.save(self.output_directory + '/y_train.npy', self.train_y)
        #np.save(self.output_directory + '/X_valid.npy', self.valid_x)
        #np.save(self.output_directory + '/y_valid.npy', self.valid_y)
        #############################
        
        ###########################################################
        #
        # added! start sampling!
        #
        ###########################################################
        """
        # undersample , old pandas version
        if count_class_0 > count_class_1: # 0 is majority
            class_0_under = class_0.sample(count_class_1, random_state=21) # for reproducibility
            data.frame = pd.concat([class_0_under, class_1], axis=0)
        else: # 1 is majority
            class_1_under = class_1.sample(count_class_0, random_state=21)
            data.frame = pd.concat([class_1_under, class_0], axis=0)
        """
        
        # undersampling
        """
        rus = RandomUnderSampler(random_state=42)
        self.train_x, self.train_y = rus.fit_resample(self.train_x, self.train_y)
        """
        
        ####################################################
        # near miss undersampling
        ###################################################
        """
        # undersample near miss 1
        undersample = NearMiss(version=1, n_neighbors=3)
        self.train_x, self.train_y = undersample.fit_resample(self.train_x, self.train_y)
        print(self.train_x.shape)
        """
            
        """
        # undersample near miss 3
        undersample = NearMiss(version=3, n_neighbors=3)
        self.train_x, self.train_y = undersample.fit_resample(self.train_x, self.train_y)
        print(self.train_x.shape)
        """
        
        """
        # distant method
        self.train_x, self.train_y = distant_method(self.train_x, self.train_y)
        """
        
        # oversample
        """
        # old pandas version
        if count_class_0 > count_class_1: # 0 is majority
            class_1_over = class_1.sample(count_class_0, random_state=21, replace=True) # for reproducibility
            data.frame = pd.concat([class_1_over, class_0], axis=0)
        else: # 1 is majority
            class_0_over = class_0.sample(count_class_1, random_state=21, replace=True)
            #print(class_0.shape[0])
            data.frame = pd.concat([class_1, class_0_over], axis=0)
        print(data.frame.shape[0])
        """
        """
        # replicated oversampling
        self.train_x, self.train_y = replicated_oversampling(self.train_x, self.train_y)
        """
        
        """
        # random oversampling
        ros = RandomOverSampler(random_state=42)
        self.train_x, self.train_y = ros.fit_resample(self.train_x, self.train_y)
        """
        # smote oversampling
        #self.train_x, self.train_y = SMOTE().fit_resample(self.train_x, self.train_y)
        
        # ADASYN oversampling
        #self.train_x, self.train_y = ADASYN().fit_resample(self.train_x, self.train_y)
        
        # gamma
        """
        self.train_x, self.train_y = gamma_oversampling(self.train_x, self.train_y)
        print("gamma", self.train_x.shape, self.train_y.shape)
        """
        
        ####################################################
        # end sampling
        ###################################################
        
        # adding different units of black
        #self.train_x, self.train_y = duplicate_black(4, self.train_x, self.train_y)
        #self.train_x, self.train_y = duplicate_30_40(20, self.train_x, self.train_y)
        #self.train_x, self.train_y = duplicate_black_minority(20, self.train_x, self.train_y)
        #self.train_x, self.train_y = duplicate_30_40_minority(7, self.train_x, self.train_y)
        
        #self.train_x, self.train_y = duplicate_asian_minority(3, self.train_x, self.train_y)
        #self.train_x, self.train_y = duplicate_40_50_minority(20, self.train_x, self.train_y)
        
        #self.train_x, self.train_y = duplicate_white_minority(20, self.train_x, self.train_y)
        #self.train_x, self.train_y = duplicate_hispanic_minority(20, self.train_x, self.train_y)
        
        #self.train_x, self.train_y = duplicate_age_minority(20, self.train_x, self.train_y)
		
		#self.train_x, self.train_y = duplicate_gender_minority(2, self.train_x, self.train_y)

        # combined DP
        #self.train_x, self.train_y = combined_DP('gamma', self.train_x, self.train_y)
		
		# new special cross-group dp
        #self.train_x, self.train_y = cross_group_DP('Black', 'random', input_columns, 18, self.train_x, self.train_y)
                               
        ######################################################
        # added for outputing training and validating set data count
        ######################################################
        
        self.train_1 = np.count_nonzero(self.train_y)
        self.train_0 = self.train_y.shape[0] - self.train_1
        self.valid_1 = np.count_nonzero(self.valid_y)
        self.valid_0 = self.valid_y.shape[0] - self.valid_1
        
        # end data count
		
        self.train_x = remove_race_features(self.train_x, input_columns)
                               
        ###########################################################
        # added for output test data after shuffling
        # only for one hot encoding
        ###########################################################
        # save before normalizing age!!! T-T
        
        #self.train_output_df = pd.DataFrame(self.train_x, columns=input_columns)
        
        self.test_output_df = pd.DataFrame(self.valid_x, columns=input_columns)
        
        # one hot encoding
        useful_columns = ['Race recode Y 1', 'Race recode Y 2', 'Race recode Y 4',\
        'Origin Recode NHIA 1',
        'Age at diagnosis continuous', 'Sex 1']
        """
        # subgroup
        useful_columns = ['Race recode Y', \
        'Origin Recode NHIA',
        'Age at diagnosis', 'Sex']
        """
        self.test_output_df = self.test_output_df[useful_columns]
        #self.train_output_df = self.train_output_df[useful_columns]
        #self.train_output_df['true y'] = self.train_y
        
        #test_output_df['predict y'] = predict_y
        #test_output_df.to_csv(self.output_directory + 'test_results_data_frame.csv')
        
        ###########################################################
        # end added code
        ###########################################################
        
        # Unique hash for set splits to ensure that same sets are used throughout experiments
        self.set_split_hash = hash(np.sum(self.train_x)) + hash(np.sum(self.train_y)) + hash(np.sum(self.valid_x)) \
                              + hash(np.sum(self.valid_y)) + hash(np.sum(self.test_x)) + hash(np.sum(self.test_y))

        # Normalize data
        if encode_categorical_inputs:
            # Only normalize continuous fields
            continuous_columns = [idc for idc, c in enumerate(list(data.frame)) if c.endswith(' continuous')]
            scaler = preprocessing.StandardScaler().fit(self.train_x[:, continuous_columns])
            self.train_x[:, continuous_columns] = scaler.transform(self.train_x[:, continuous_columns])
            self.valid_x[:, continuous_columns] = scaler.transform(self.valid_x[:, continuous_columns])
            self.test_x[:, continuous_columns] = scaler.transform(self.test_x[:, continuous_columns])
        else:
            # Normalize all fields
            scaler = preprocessing.StandardScaler().fit(self.train_x)
            self.train_x = scaler.transform(self.train_x)
            self.valid_x = scaler.transform(self.valid_x)
            self.test_x = scaler.transform(self.test_x)

        logging.info("Data:  " + str(data.frame.shape) + " -> x:" + str(x.shape) + ", y:" + str(y.shape))
        logging.info("Train: x:{0}, y:{1}".format(str(self.train_x.shape), str(self.train_y.shape)))
        logging.info("Valid: x:{0}, y:{1}".format(str(self.valid_x.shape), str(self.valid_y.shape)))
        logging.info("Test:  x:{0}, y:{1}".format(str(self.test_x.shape), str(self.test_y.shape)))

        # Split up the data into distinct inputs for each embedding
        if model_type == 'MLPEmb':
            print("")
            logging.info("Embed input data.")
            encoding_splits = np.cumsum(list(encodings.values()))
            self.train_x = np.hsplit(self.train_x, encoding_splits)[:-1]
            self.train_x = [np.expand_dims(x, axis=2) for x in self.train_x]
            self.valid_x = np.hsplit(self.valid_x, encoding_splits)[:-1]
            self.valid_x = [np.expand_dims(x, axis=2) for x in self.valid_x]
            self.test_x = np.hsplit(self.test_x, encoding_splits)[:-1]
            self.test_x = [np.expand_dims(x, axis=2) for x in self.test_x]

    def train(self, mlp_epochs):
        """ Training procedure. """
        mlp_batch_size = 20

        if self.model_type in ['MLP', 'MLPEmb']:
            self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                 validation_data=(self.valid_x, self.valid_y))
            ######################################
            # save this model
            #self.model.model.save(self.output_directory + 'breast_whole_mlp.h5')
            ######################################
            """
            # extra training on black
            train_x_b = np.load('data/X_train_b.npy')
            train_y_b = np.load('data/y_train_b.npy')
            train_x_b = np.repeat(train_x_b, 10, axis=0)
            train_y_b = np.repeat(train_y_b, 10, axis=0)
            train_x_b, train_y_b = shuffle(train_x_b, train_y_b, random_state=0)
            self.model.model.fit(train_x_b, train_y_b, epochs=1)
            """
        elif self.model_type in ['LogR', 'LinR', 'SVM', 'NAIVE']:
            self.model.model.fit(self.train_x, self.train_y)

    def validate(self):
        """ Validation evaluation wrapper. """
        print('Validation results: ', end='')
        return self.evaluate(self.valid_x, self.valid_y)

    def test(self):
        """ Testing evaluation wrapper. """
        print('Test results: ', end='')
        return self.evaluate(self.test_x, self.test_y)

    def evaluate(self, eval_x, eval_y):
        """ Generic evaluation method. """
        ######################################
        # save this model
        #self.model.model.save(self.output_directory + 'breast_whole_mlp')
        ######################################

        if self.task in ['survival12', 'survival60'] and (self.model_type == 'SVM' or self.model_type == 'LogR'):
            # Use decision function value as score
            # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
            scores_y = self.model.model.decision_function(eval_x)
        else:
            scores_y = self.model.model.predict(eval_x)

        measurements = []
        # Regression
        if self.task in ['mort12', 'mort60']:
            n = float(self.task[-2:])
            scaled_eval_y = eval_y * n
            scaled_scores_y = scores_y * n

            measurements.append('rmse = ' + str(np.sqrt(mean_squared_error(eval_y, scores_y))))
            measurements.append('srmse = ' + str(np.sqrt(mean_squared_error(scaled_eval_y, scaled_scores_y))))
            measurements.append('smae = ' + str(mean_absolute_error(scaled_eval_y, scaled_scores_y)))

            if self.plot_results:
                fig = plt.figure(dpi=200)
                self.plot_scatter(scaled_eval_y, scaled_scores_y, plt)
                fig.savefig(self.output_directory + 'scatter.png')

        # Classification
        elif self.task in ['survival12', 'survival60']:
            if self.model_type == 'SVM' or self.model_type == 'LogR':
                predict_y = self.model.model.predict(eval_x)
            else:
                predict_y = scores_y.round()
                
            ###########################################################
            # added for output test data after shuffling
            # only for one hot encoding
            ###########################################################
            # append true and predict labels and save to file
            # for subgroup training
            """
            self.test_output_df = pd.DataFrame(eval_y, columns=['true y'])
            self.test_output_df['predict y'] = predict_y
            self.test_output_df['score y'] = scores_y
            # for whole group training
            """
            self.test_output_df['true y'] = eval_y
            self.test_output_df['predict y'] = predict_y
            self.test_output_df['score y'] = scores_y
            if self.model_type == 'LogR':
                self.test_output_df['predict y proba'] = self.model.model.predict_proba(eval_x)[:,1]
            
            self.test_output_df.to_csv(self.output_directory + 'test_results_data_frame.csv')
            
            #self.train_output_df.to_csv(self.output_directory + 'train_data_frame.csv')
        
            ###########################################################
            # end added code
            ###########################################################

            measurements.append('auc = ' + str(roc_auc_score(eval_y, scores_y)))
            measurements.append('f1 = ' + str(f1_score(eval_y, predict_y)))
            measurements.append('acc = ' + str(accuracy_score(eval_y, predict_y)))
            ################################################
            #---added------------------------------------
            measurements.append('balanced_accuracy = ' + str(balanced_accuracy_score(eval_y, predict_y)))
            measurements.append('precision class 1 = ' + str(precision_score(eval_y, predict_y)))
            measurements.append('recall class 1 = ' + str(recall_score(eval_y, predict_y)))
            tn, fp, fn, tp = confusion_matrix(eval_y, predict_y).ravel()
            recall0 = tn / (tn + fp)
            precision0 = tn / (tn + fn)
            measurements.append('f1 class 0 = ' + str(2 * precision0 * recall0 / (precision0 + recall0)))
            measurements.append('precision class 0 = ' + str(precision0))
            measurements.append('recall class 0 = ' + str(recall0))
            measurements.append('FPR = ' + str(fp / (fp + tn)))
            measurements.append('FNR = ' + str(fn / (fn + tp)))
            ################################################
            

            if self.plot_results:
                fig = plt.figure(dpi=200)
                self.plot_roc(eval_y, scores_y, plt)
                fig.savefig(self.output_directory + 'roc.png')

        print(', '.join(measurements))
        return measurements

    def importance(self, encodings):
        """ Method that analyzes the importance of input variables for LogR/LinR and MLP* models. """
        importance = []

        if self.model_type in ['LogR', 'LinR'] and self.task in ['survival12', 'survival60']:
            
            # Use coefficients
            abs_coefficients = np.abs(self.model.model.coef_[0])
            i = 0
            for column, encoding_size in encodings.items():
                coefficient_sum = 0.
                for idx in range(i, i + encoding_size):
                    coefficient_sum += abs_coefficients[idx]
                i += encoding_size
                importance.append(coefficient_sum / encoding_size) # sum or average???
            importance = np.array(importance)
            """
            # Ablate attributes and measure effect on output
            scores_y = self.model.model.predict_proba(self.test_x)[:,1]
            i = 0
            for column, encoding_size in encodings.items():
                ablated_test_x = self.test_x.copy()

                ablated_test_x[:, i:(i + encoding_size)] = 0
                i += encoding_size
                
                ablated_scores_y = self.model.model.predict_proba(ablated_test_x)[:,1]
                ablated_diff = np.sum(np.abs(scores_y - ablated_scores_y))
                importance.append(ablated_diff)
            """

        if self.model_type in ['MLP', 'MLPEmb'] and self.task in ['survival12', 'survival60']:
            # Ablate attributes and measure effect on output
            scores_y = self.model.model.predict(self.test_x)
            i = 0
            for column, encoding_size in encodings.items():
                ablated_test_x = self.test_x.copy()
                if self.model_type == 'MLP':
                    ablated_test_x[:, i:(i + encoding_size)] = 0
                    i += encoding_size
                elif self.model_type == 'MLPEmb':
                    ablated_test_x[i][:, :] = 0
                    i += 1
                ablated_scores_y = self.model.model.predict(ablated_test_x)
                ablated_diff = np.sum(np.abs(scores_y - ablated_scores_y))
                importance.append(ablated_diff)

            importance = np.array(importance)

        # Normalize importance
        importance = importance / np.sum(importance)
        result = dict(zip([column for column, encoding_size in encodings.items()], importance))

        # Sort results
        result = [(k, result[k]) for k in sorted(result, key=result.get, reverse=True)]
        return result

    @staticmethod
    def plot_scatter(labels, predictions, plot):
        """ Method to plot a scatter plot of predictions vs labels """
        plot.scatter(labels, predictions)
        plot.xlabel('Labels')
        plot.ylabel('Predictions')
        plot.title('Labels vs predictions')

    @staticmethod
    def plot_roc(labels, scores, plot):
        """ Method to plot ROC curve from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plot.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plot.xlim([0.0, 1.0])
        plot.ylim([0.0, 1.05])
        plot.xlabel('False Positive Rate')
        plot.ylabel('True Positive Rate')
        plot.title('Receiver operating characteristic example')
        plot.legend(loc="lower right")
