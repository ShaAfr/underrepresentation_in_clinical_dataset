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
    for m in majority:
        try:
            dist, ind = nn.kneighbors(m.reshape(1, -1), 3, return_distance=True)
            ave = np.mean(dist)
        except NotFittedError:
            ave = 0
            count += 1
            
        distance.append(ave)
    
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
    
    
def duplicate_black(units, X, y):
    # if units == 0?
    if units == 0.25:
        X_new = []
        y_new = []
        count = 0
        for i in range(len(X)):
            if X[i][726] == 1: # black
                count+=1
                if count == 4: # select 1 in 4
                    X_new.append(X[i])
                    y_new.append(y[i])
                    count = 0
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    elif units == 0.5:
        X_new = []
        y_new = []
        selected = False
        for i in range(len(X)):
            if X[i][726] == 1: # black
                if selected: # last one selected, skip this one
                    selected = False
                else:
                    X_new.append(X[i])
                    y_new.append(y[i])
                    selected = True
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    elif units == 0.75:
        X_new = []
        y_new = []
        count = 0
        for i in range(len(X)):
            if X[i][726] == 1: # black
                count+=1
                if count == 4: # skip 1 in 4
                    count = 0
                else:
                    X_new.append(X[i])
                    y_new.append(y[i])
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    else: # a number > 1
        train_x_b = np.load('data/lung/X_train_b.npy')
        train_y_b = np.load('data/lung/y_train_b.npy')
        train_x_b = np.repeat(train_x_b, units - 1, axis=0)
        train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
        X = np.concatenate((X, train_x_b), axis=0)
        y = np.concatenate((y, train_y_b), axis=0)
        
        # random_state=0 originally
        X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_30_40(units, X, y):
    if units == 0.25:
        X_new = []
        y_new = []
        count = 0
        for i in range(len(X)):
            if X[i][0] >= 30 and X[i][0] < 40: # 30-40
                count+=1
                if count == 4: # select 1 in 4
                    X_new.append(X[i])
                    y_new.append(y[i])
                    count = 0
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    elif units == 0.5:
        X_new = []
        y_new = []
        selected = False
        for i in range(len(X)):
            if X[i][0] >= 30 and X[i][0] < 40: # black
                if selected: # last one selected, skip this one
                    selected = False
                else:
                    X_new.append(X[i])
                    y_new.append(y[i])
                    selected = True
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    elif units == 0.75:
        X_new = []
        y_new = []
        count = 0
        for i in range(len(X)):
            if X[i][0] >= 30 and X[i][0] < 40: # black
                count+=1
                if count == 4: # skip 1 in 4
                    count = 0
                else:
                    X_new.append(X[i])
                    y_new.append(y[i])
            else:
                X_new.append(X[i])
                y_new.append(y[i])
        X = np.array(X_new)
        y = np.array(y_new)
    else: # a number > 1
        train_x_b = np.load('data/lung/X_train_80_90.npy')
        train_y_b = np.load('data/lung/y_train_80_90.npy')
        train_x_b = np.repeat(train_x_b, units - 1, axis=0)
        train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
        X = np.concatenate((X, train_x_b), axis=0)
        y = np.concatenate((y, train_y_b), axis=0)
        
        X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_black_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_b_minor.npy')
    train_y_b = np.load('data/breast/y_train_b_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_30_40_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_30_40_minor.npy')
    train_y_b = np.load('data/breast/y_train_30_40_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_asian_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_a_minor.npy')
    train_y_b = np.load('data/breast/y_train_a_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_white_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_w_minor.npy')
    train_y_b = np.load('data/breast/y_train_w_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_hispanic_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_h_minor.npy')
    train_y_b = np.load('data/breast/y_train_h_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
    
def duplicate_40_50_minority(units, X, y):
    #train_x_b = np.load('data/breast/X_train_40_50_minor.npy')
    #train_y_b = np.load('data/breast/y_train_40_50_minor.npy')
    train_x_b = np.load('data/lung/X_train_70_80_minor.npy')
    train_y_b = np.load('data/lung/y_train_70_80_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
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
    
    generated = []
    
    for m in minority_to_replace:
        gamma_t = np.random.gamma(gamma_shape_alpha, gamma_scale_theta, 1)[0]
        
        neighbors = nn.kneighbors(minority[m].reshape(1, -1), return_distance=False)
        #print(neighbors)
        neigh_idx = neighbors[0][random.randint(0, 3)] # k = 4
        v = minority[neigh_idx] - minority[m]
        new_point = minority[m] + (gamma_t - gamma_max_value) * v
        generated.append(new_point)
        
    generated = np.array(generated)
    X_bal = np.concatenate((X, generated), axis=0)
    
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)
    
    return X_bal, y_bal
        
        
def combined_DP(mtd, X, y):
    # breast black
    f = open('data/breast/input_columns.txt', 'r')
    cols = f.readlines()
    group_ind = cols.index('Race recode Y 2\n')
    race = X[:, group_ind]
    ind = np.where(race == 0) # everything not in this group
    X_ = X[ind]
    y_ = y[ind]
    
    train_x_b = np.load('data/breast/X_train_b.npy')
    train_y_b = np.load('data/breast/y_train_b.npy')
    
    if mtd == 'SMOTE':
        train_x_b, train_y_b = SMOTE().fit_resample(train_x_b, train_y_b)
    elif mtd == 'ADASYN':
        train_x_b, train_y_b = ADASYN().fit_resample(train_x_b, train_y_b)
    else: #gamma
        train_x_b, train_y_b = gamma_oversampling(train_x_b, train_y_b)
        
    X_bal = np.concatenate((X_, train_x_b), axis=0)
    y_bal = np.concatenate((y_, train_y_b), axis=0)
    
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=0)
    
    return X_bal, y_bal
    
    
def duplicate_age_minority(units, X, y):
    train_x_b = np.load('data/lung/X_train_below_30_minor.npy')
    train_y_b = np.load('data/lung/y_train_below_30_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y

def duplicate_gender_minority(units, X, y):
    train_x_b = np.load('data/breast/X_train_m_minor.npy')
    train_y_b = np.load('data/breast/y_train_m_minor.npy')
    train_x_b = np.repeat(train_x_b, units - 1, axis=0)
    train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
    X = np.concatenate((X, train_x_b), axis=0)
    y = np.concatenate((y, train_y_b), axis=0)
        
    X, y = shuffle(X, y, random_state=0)
    
    return X, y
	
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

	