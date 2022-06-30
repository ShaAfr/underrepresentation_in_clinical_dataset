import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle
import random
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

def distant_method(X, y):
# X, y as 2 np arrays

    # count the number of 2 classes
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1

    zeros = np.where(y == 0)[0] # indices of class 0 cases
    ones = np.where(y == 1)[0] # indices of class 1 cases
    if count_class_0 > count_class_1: # 0 is majority
        majority = X[zeros]
        minority = X[ones]
        num_cases = count_class_1
        y_bal = np.concatenate((np.zeros(num_cases), np.ones(num_cases)), axis=0).reshape((2*num_cases,)).astype(np.int32)
    else:
        majority = X[ones]
        minority = X[zeros]
        num_cases = count_class_0
        y_bal = np.concatenate((np.ones(num_cases), np.zeros(num_cases)), axis=0).reshape((2*num_cases,)).astype(np.int32)

    # get average distance to 3 nearest minority cases
    nn = NearestNeighbors()
    nn.fit(minority)
    distance = []
    count = 0
    print("distant len majority", len(majority))
    for m in majority:
        try:
            dist, ind = nn.kneighbors(m.reshape(1, -1), 3, return_distance=True)
            ave = np.mean(dist)
        except NotFittedError:
            ave = 0
            count += 1

        distance.append(ave)

        if len(distance) % 5000 == 0:
            print(len(distance))
    print(count)

    # sort
    distance = np.array(distance)
    sorted_ind = np.argsort(distance)

    # pick farthest
    selected = majority[list(sorted_ind)[-num_cases:]]

    X_bal = np.concatenate((selected, minority), axis=0)

    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)

    return X_bal, y_bal

def replicated_oversampling(X, y):
    # count the number of 2 classes
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1

    zeros = np.where(y == 0)[0] # indices of class 0 cases
    ones = np.where(y == 1)[0] # indices of class 1 cases
    if count_class_0 > count_class_1: # 0 is majority
        majority = X[zeros]
        minority = X[ones]
        num_cases = count_class_1
        multi = int(round(count_class_0 / count_class_1))
        y_bal = np.concatenate((np.zeros(count_class_0), np.ones(count_class_1 * multi)), axis=0) \
            .reshape((count_class_0 + count_class_1 * multi,)).astype(np.int32)
    else:
        majority = X[ones]
        minority = X[zeros]
        num_cases = count_class_0
        multi = int(round(count_class_1 / count_class_0))
        y_bal = np.concatenate((np.ones(count_class_1), np.zeros(count_class_0 * multi)), axis=0) \
            .reshape((count_class_1 + count_class_0 * multi,)).astype(np.int32)

    minority = np.repeat(minority, multi, axis=0)

    X_bal = np.concatenate((majority, minority), axis=0)

    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)

    return X_bal, y_bal


def double_prioritize(X, y, units, subgroup, cols):
    """
    units: number of units of minority to be included in the input data, (units - 1) additional units will be added
    subgroup: the subgroup to be priorized using dp, determined by different columns in the input data
    cols: columns of input dataset, useful for finding the column specifying gender, race, or age
    """
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1

    minority_class = int(count_class_1 < count_class_0)

    # subgroup index in input data
    if subgroup == 'Male':
        ind = np.where(X[:,cols.index('Sex 1')] == 1)
    elif subgroup == 'Female':
        ind = np.where(X[:,cols.index('Sex 1')] == 0)
    elif subgroup == 'White':
        ind = np.where(X[:,cols.index('Race recode Y 1')] == 1)
    elif subgroup == 'Black':
        ind = np.where(X[:,cols.index('Race recode Y 2')] == 1)
    elif subgroup == 'Hispanic':
        ind = np.where(X[:,cols.index('Origin Recode NHIA 1')] == 1)
    elif subgroup == 'Asian':
        ind = np.where(X[:,cols.index('Race recode Y 4')] == 1)
    elif subgroup == 'Age30':
        ind = np.where(X[:,cols.index('Age at diagnosis continuous')] < 30)
    elif subgroup == '3040':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 30) & (X[:,cols.index('Age at diagnosis continuous')] < 40))
    elif subgroup == '4050':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 40) & (X[:,cols.index('Age at diagnosis continuous')] < 50))
    elif subgroup == '5060':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 50) & (X[:,cols.index('Age at diagnosis continuous')] < 60))
    elif subgroup == '6070':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 60) & (X[:,cols.index('Age at diagnosis continuous')] < 70))
    elif subgroup == '7080':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 70) & (X[:,cols.index('Age at diagnosis continuous')] < 80))
    elif subgroup == '8090':
        ind = np.where((X[:,cols.index('Age at diagnosis continuous')] >= 80) & (X[:,cols.index('Age at diagnosis continuous')] < 90))
    elif subgroup == 'Age90':
        ind = np.where(X[:,cols.index('Age at diagnosis continuous')] >= 90)
    else:
        logging.error('Invalid subgroup for DP.')
        exit(-1)

    X_sub = X[ind]
    y_sub = y[ind]
    #unit_size = y_sub.shape[0]

    # minority samples in the selected subgroup
    ind_minor = np.where(y_sub == minority_class)
    X_sub = X_sub[ind_minor]
    y_sub = y_sub[ind_minor]

    """
    X_sub = np.repeat(X_sub, units - 1, axis=0)
    y_sub = np.repeat(y_sub, units - 1, axis=0)
    print('DP: Added ' + str(units-1) + ' additional units ' + subgroup + ' minority data (' + str(y_sub.shape[0]) + ' samples).')

    X = np.concatenate((X, X_sub), axis=0)
    y = np.concatenate((y, y_sub), axis=0)
    """
    for i in range(units-1):
        X = np.concatenate((X, X_sub), axis=0)
        y = np.concatenate((y, y_sub), axis=0)
        #print('np.concatenate ' + str(i) + ' units')
    print('DP: Added ' + str(units-1) + ' additional units ' + subgroup + ' minority data (' + str(y_sub.shape[0] * (units-1)) + ' samples).')
    X, y = shuffle(X, y, random_state=0)

    return X, y

def combined_double_prioritize(X, y, units, subgroup, cols, method):
    """
    units: number of units of minority to be included in the input data, (units - 1) additional units will be added
    subgroup: the subgroup to be priorized using dp, determined by different columns in the input data
    cols: columns of input dataset, useful for finding the column specifying gender, race, or age
    method: which sampling method to use. plain DP just replicate the samples. here different methods can be chosen.
    'ADASYN', 'SMOTE', 'Gamma' as possible options.
    """
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1

    minority_class = int(count_class_1 < count_class_0)
    #majority_class = int(count_class_1 > count_class_0)

    # subgroup index in input data
    if subgroup == 'Male':
        ind = np.where(X[:,cols.index('Sex 1')] == 1)
        ind_other = np.where(X[:,cols.index('Sex 1')] == 0)
    elif subgroup == 'Female':
        ind = np.where(X[:,cols.index('Sex 1')] == 0)
        ind_other = np.where(X[:,cols.index('Sex 1')] == 1)
    elif subgroup == 'White':
        ind = np.where(X[:,cols.index('Race recode Y 1')] == 1)
        ind_other = np.where(X[:,cols.index('Race recode Y 1')] == 0)
    elif subgroup == 'Black':
        ind = np.where(X[:,cols.index('Race recode Y 2')] == 1)
        ind_other = np.where(X[:,cols.index('Race recode Y 2')] == 0)
    elif subgroup == 'Hispanic':
        ind = np.where(X[:,cols.index('Origin Recode NHIA 1')] == 1)
        ind_other = np.where(X[:,cols.index('Origin Recode NHIA 1')] == 0)
    elif subgroup == 'Asian':
        ind = np.where(X[:,cols.index('Race recode Y 4')] == 1)
        ind_other = np.where(X[:,cols.index('Race recode Y 4')] == 0)
    elif subgroup == 'Age30':
        ind = np.where(X[:,cols.index('Age at diagnosis continuous')] < 30)
        ind_other = np.where(X[:,cols.index('Age at diagnosis continuous')] >= 30)
    elif subgroup == '3040':
        ind = np.where(30 <= X[:,cols.index('Age at diagnosis continuous')] < 40)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 30) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 40)))
    elif subgroup == '4050':
        ind = np.where(40 <= X[:,cols.index('Age at diagnosis continuous')] < 50)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 40) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 50)))
    elif subgroup == '5060':
        ind = np.where(50 <= X[:,cols.index('Age at diagnosis continuous')] < 60)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 50) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 60)))
    elif subgroup == '6070':
        ind = np.where(60 <= X[:,cols.index('Age at diagnosis continuous')] < 70)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 60) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 70)))
    elif subgroup == '7080':
        ind = np.where(70 <= X[:,cols.index('Age at diagnosis continuous')] < 80)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 70) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 80)))
    elif subgroup == '8090':
        ind = np.where(80 <= X[:,cols.index('Age at diagnosis continuous')] < 90)
        ind_other = np.where((X[:,cols.index('Age at diagnosis continuous')] < 80) | \
                            ((X[:,cols.index('Age at diagnosis continuous')] >= 90)))
    elif subgroup == 'Age90':
        ind = np.where(X[:,cols.index('Age at diagnosis continuous')] >= 90)
        ind_other = np.where(X[:,cols.index('Age at diagnosis continuous')] < 90)
    else:
        logging.error('Invalid subgroup for DP.')
        exit(-1)

    X_sub = X[ind]
    y_sub = y[ind]

    # minority samples in the selected subgroup
    ind_minor = np.where(y_sub == minority_class)
    y_sub_minor = y_sub[ind_minor]

    # passing in a ratio to oversampling methods
    if method == 'SMOTE':
        smote = SMOTE(sampling_strategy={minority_class: y_sub_minor.shape[0] * units})
        X_sub, y_sub = smote.fit_resample(X_sub, y_sub)
        print('dp smote fit')
    elif method == 'ADASYN':
        adasyn = ADASYN(sampling_strategy={minority_class: y_sub_minor.shape[0] * units})
        X_sub, y_sub = adasyn.fit_resample(X_sub, y_sub)
        print('dp adasyn fit')
    elif method == 'Gamma':
        X_sub, y_sub = gamma_oversampling_for_DP(X_sub, y_sub, units)
        print('dp gamma fit')

    print('DP ' + method + ': Added ' + str(units-1) + ' additional units ' + subgroup + ' minority data (' + str(y_sub_minor.shape[0] * (units-1)) + ' samples).')

    # concat other samples and the duplicated subgroup
    X = np.concatenate((X[ind_other], X_sub), axis=0)
    y = np.concatenate((y[ind_other], y_sub), axis=0)

    X, y = shuffle(X, y, random_state=0)

    return X, y


def gamma_oversampling(X, y):
    gamma_shape_alpha = 2
    gamma_scale_theta = 0.125
    gamma_max_value = 0.125

    # count the number of 2 classes
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1
    zeros = np.where(y == 0)[0] # indices of class 0 cases
    ones = np.where(y == 1)[0] # indices of class 1 cases
    if count_class_0 > count_class_1: # 0 is majority
        majority = X[zeros]
        minority = X[ones]

        amount_to_add = count_class_0 - count_class_1
        y_bal = np.concatenate((y, np.ones(amount_to_add)), axis=0).reshape((2*count_class_0,)).astype(np.int32)
    else:
        majority = X[ones]
        minority = X[zeros]
        num_cases = count_class_1

        amount_to_add = count_class_1 - count_class_0
        y_bal = np.concatenate((y, np.zeros(amount_to_add)), axis=0).reshape((2*count_class_1,)).astype(np.int32)

    nn = NearestNeighbors(n_neighbors=4)
    nn.fit(minority)


    minority_to_replace = random.choices(list(range(len(minority))), k=amount_to_add)
    print("len minor to replace", len(minority_to_replace))

    generated = []

    for m in minority_to_replace:
        gamma_t = np.random.gamma(gamma_shape_alpha, gamma_scale_theta, 1)[0]

        neighbors = nn.kneighbors(minority[m].reshape(1, -1), return_distance=False)
        #print(neighbors)
        neigh_idx = neighbors[0][random.randint(0, 3)] # k = 4
        v = minority[neigh_idx] - minority[m]
        new_point = minority[m] + (gamma_t - gamma_max_value) * v
        generated.append(new_point)
        if len(generated) % 5000 == 0:
            print(len(generated))

    generated = np.array(generated)
    X_bal = np.concatenate((X, generated), axis=0)

    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)

    return X_bal, y_bal

def gamma_oversampling_for_DP(X, y, units):
    """
    gamma oversampling for DP, where how many minority to create is given
    X, y: training data of a subgroup to apply DP
    units: units of minority in the training data at the end
    """
    gamma_shape_alpha = 2
    gamma_scale_theta = 0.125
    gamma_max_value = 0.125

    # count the number of 2 classes
    count_class_1 = np.count_nonzero(y)
    count_class_0 = y.shape[0] - count_class_1
    zeros = np.where(y == 0)[0] # indices of class 0 cases
    ones = np.where(y == 1)[0] # indices of class 1 cases
    if count_class_0 > count_class_1: # 0 is majority
        majority = X[zeros]
        minority = X[ones]

        amount_to_add = count_class_1 * (units-1)
        y_bal = np.concatenate((y, np.ones(amount_to_add)), axis=0).reshape((count_class_0 + count_class_1 * units,)).astype(np.int32)
    else:
        majority = X[ones]
        minority = X[zeros]
        num_cases = count_class_1

        amount_to_add = count_class_0 * (units-1)
        y_bal = np.concatenate((y, np.zeros(amount_to_add)), axis=0).reshape((count_class_1 + count_class_0 * units,)).astype(np.int32)

    nn = NearestNeighbors(n_neighbors=4)
    nn.fit(minority)


    minority_to_replace = random.choices(list(range(len(minority))), k=amount_to_add)
    print("len minor to replace", len(minority_to_replace))

    generated = []

    for m in minority_to_replace:
        gamma_t = np.random.gamma(gamma_shape_alpha, gamma_scale_theta, 1)[0]

        neighbors = nn.kneighbors(minority[m].reshape(1, -1), return_distance=False)
        #print(neighbors)
        neigh_idx = neighbors[0][random.randint(0, 3)] # k = 4
        v = minority[neigh_idx] - minority[m]
        new_point = minority[m] + (gamma_t - gamma_max_value) * v
        generated.append(new_point)
        if len(generated) % 5000 == 0:
            print(len(generated))

    generated = np.array(generated)
    X_bal = np.concatenate((X, generated), axis=0)

    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)

    return X_bal, y_bal

def cross_group_DP(test_group, group_to_add, cols, units, X, y):
	if group_to_add == 'Asian':
		train_x_ = np.load('data/breast/X_train_a_minor.npy')
		train_y_ = np.load('data/breast/y_train_a_minor.npy')
		print('Asian selected')
	elif group_to_add == 'Hispanic':
		train_x_ = np.load('data/breast/X_train_h_minor.npy')
		train_y_ = np.load('data/breast/y_train_h_minor.npy')
		print('Hispanic selected')
	elif group_to_add == 'Black':
		train_x_ = np.load('data/breast/X_train_b_minor.npy')
		train_y_ = np.load('data/breast/y_train_b_minor.npy')
		print('Black selected')
	else: #random
		whole_minor_ind = np.where(y == 0)
		train_x_ = X[whole_minor_ind]
		train_y_ = y[whole_minor_ind]
		print('random selected')

	if test_group == 'Asian':
		ind = np.where(X[:,cols.index('Race recode Y 4')])
	elif test_group == 'Hispanic':
		ind = np.where(X[:,cols.index('Origin Recode NHIA 1')])
	else: #black
		ind = np.where(X[:,cols.index('Race recode Y 2')])

	y_group = y[ind]
	ind_minor = np.where(y_group == 0) # 0 is minor for breast
	c1_number = len(y[ind_minor])

	number_to_add = c1_number * (units - 1)
	print(str(c1_number) + ' c1 samples ' + str(units - 1) + ' units added')

	random_ind = np.random.choice(len(train_x_), number_to_add)
	print(len(random_ind))

	X = np.concatenate((X, train_x_[random_ind]), axis=0)
	y = np.concatenate((y, train_y_[random_ind]), axis=0)

	X, y = shuffle(X, y, random_state=0)

	return X, y

def remove_race_features(X_train, cols):
	cols = np.array(cols)
	#race_ind = np.where(('Race' in cols) | ('Origin' in cols))
	mask = [('Race' in x) or ('Origin' in x) for x in cols]
	mask = np.array(mask)
	race_ind = np.where(mask)
	print(race_ind)
	X_train[:,race_ind] = 0
	#X_test[:,race_ind] = 0

	return X_train

def remove_irr_features(X_train, cols):
	cols = np.array(cols)
	to_remove = []
	mask = [('SEER registry' in x) or ('Month of diagnosis' in x) or ('State-county recode' in x) or ('Type of reporting source' in x) for x in cols]
	mask = np.array(mask)
	removed_ind = np.where(mask)
	print(removed_ind)
	X_train[:,removed_ind] = 0

	return X_train
