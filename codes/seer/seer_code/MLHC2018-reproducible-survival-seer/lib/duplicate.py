


def duplicate(units, X, y):
    # if units == 0?
    if units == 0.5:
        # delete every other cases
        deleted = False
        for i in range(len(X)):
            if X[i][726] == 1: # black
                if deleted: # last one deleted, keep this one
                    deleted = False
                else:
                    np.delete(X, i, 0)
                    np.delete(y, i, 0)
                    i -= 1
                    deleted = True
            
    else: # a number > 1
        train_x_b = np.load('data/X_train_b.npy')
        train_y_b = np.load('data/y_train_b.npy')
        train_x_b = np.repeat(train_x_b, units - 1, axis=0)
        train_y_b = np.repeat(train_y_b, units - 1, axis=0)
        
        X = np.concatenate((X, train_x_b), axis=0)
        y = np.concatenate((y, train_y_b), axis=0)
        
        X, y = shuffle(X, y, random_state=0)
    
    return X, y