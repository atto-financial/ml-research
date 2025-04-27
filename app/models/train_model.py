from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def data_split(raw_dat):
    X = raw_dat.drop(["cdd1","cdd2","cdd3","cdd4","cdd5","cdd6","cdd7","cdd8","cdd11","ust", "ins"], axis=1)
    y = raw_dat["ust"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
