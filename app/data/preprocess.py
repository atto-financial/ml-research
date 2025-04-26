from sklearn.model_selection import train_test_split

def preprocess_data(raw_dat):
    X = raw_dat.drop(["ust", "ins"], axis=1)
    y = raw_dat["ust"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y


def preprocess_data_2(raw_dat):
    X = raw_dat.drop(["ust", "ins"], axis=4)
    y = raw_dat["ust"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X, y