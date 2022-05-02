import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # true, pred



class Utils:

    @staticmethod
    def load_data(path_to_dataset, path_to_metadata):
        df_Rs = pd.read_csv(path_to_dataset, usecols=['filename', 'R'])
        df_metadata = pd.read_csv(path_to_metadata)
        data = pd.merge(df_Rs, df_metadata, on=["filename"])
        return data
    
    @staticmethod
    def serialize_and_save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model written to: output/{filename}")

    @staticmethod
    def create_polynomial_model():
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
        return model


    @staticmethod
    def create_ensemble_model():
        from sklearn.linear_model import LinearRegression, RidgeCV
        from sklearn.svm import SVR
        from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor

        gb = GradientBoostingRegressor(random_state=1, n_estimators=100, learning_rate=0.1, max_depth=1)
        rf = RandomForestRegressor(random_state=1, n_estimators=10, max_features=1, max_leaf_nodes=5)
        lr = LinearRegression()
        ridge = RidgeCV()
        svr = SVR()

        regressors = VotingRegressor(estimators=[\
            ('gb', gb), \
            ('rf', rf), \
            ('lr', lr), \
            ('ridge', ridge), \
            ('svm', svr)\
        ])

        model = make_pipeline(StandardScaler(), regressors)
        return model


def run_model_training(args):
    print("\n\nTraining model...\n\n")
    
    # Init arguments from cli or config file
    random_state = 66
    path_to_dataset = args.get('dataset')
    path_to_metadata = args.get('metadata')
    model_name = args.get('modelname')
    use_polynomial_model = args.get('use_polynomial_model')

    # Load dataset and run train/test 70/30 split
    data = Utils.load_data(path_to_dataset, path_to_metadata)
    X, y = data.iloc[:,1].values.reshape(-1, 1), data.iloc[:,-1].ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    print("Train size: ", len(X_train), "\nTest size: ", len(X_test))

    # Create Model and run cross validation
    if(use_polynomial_model):
        model = Utils.create_polynomial_model()
    else:
        model = Utils.create_ensemble_model()
    
    scoring = ("r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error")
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    df_scores = pd.DataFrame(scores)
    print("\nCross Validation on Train Set:")
    print(df_scores.head(5))

    # Fit the model
    model = model.fit(X_train, y_train)
    # Test model on test set
    y_pred = model.predict(X_test)
    r2, mae, mse = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred)
    print("\nTest Scores:")
    print(f"R2 = {r2} \nMAE = {mae} \nMSE = {mse} \nRMSE = {np.sqrt(mse)}\n")
    
    # Serialize and save model
    Path("models").mkdir(parents=True, exist_ok=True)
    Utils.serialize_and_save_model(model, f"models/{model_name}")
    

if __name__ == "__main__":
    print(f"\n\nYou've executed this program incorrectly. Please run `python main.py train` \n")

