import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

def train(file_name):
    df = pd.read_csv(file_name)
    onehotencoder = OneHotEncoder(handle_unknown='ignore')

    columns_to_encode = ['defect_type', 'defect_location', 'inspection_method']
        
    for column in columns_to_encode:
        onehotencoder.fit(df[column].values.reshape(-1, 1))  
        X = onehotencoder.transform(df[column].values.reshape(-1, 1)).toarray()
        encoded_defect_type = pd.DataFrame(X, columns=onehotencoder.categories_[0])
        df = pd.concat([df, encoded_defect_type], axis=1)
        df = df.drop([column], axis=1)
        with open(f"{column}_encoder.pkl", "wb") as f:
            pickle.dump(onehotencoder, f)


    label_encoder = preprocessing.LabelEncoder()
    df['severity'] = label_encoder.fit_transform(df['severity'])
    with open("severity_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    X = df[['product_id', 'severity', 'Cosmetic', 'Functional', 'Structural', 'Component', 'Internal', 'Surface', 'Automated Testing', 'Manual Testing', 'Visual Inspection']]  
    y = df['repair_cost']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {
        'n_estimators': 30,  
        'learning_rate': 0.01, 
        'max_depth': 3,  
        'subsample': 1.0, 
        'min_samples_split': 7,  
        'min_samples_leaf': 1,  
        'max_features': 'sqrt',  
        }

    model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    with open('GB_Boost_Classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return mse

