from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import os

class LogRegression():

    def __init__(self):
        self.clf = LogisticRegression()

    def save(self, save_dir):

        save_path = os.path.join(save_dir, "model.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.clf, f)
        print(f"Model saved to {save_path}")

    def fit(self, train_loader, val_loader):
        X = train_loader.drop(columns=["label"]).to_numpy()
        y = np.array(train_loader["label"])
        self.clf.fit(X, y)
        
    def predict(self, data_loader, return_prob=False):
        X = data_loader.drop(columns=["label"]).to_numpy()
        y_true = np.array(data_loader["label"])

        if return_prob:
            y_prob = self.clf.predict_proba(X)[:, 1]
            return y_true, y_prob
        else:
            y_pred = self.clf.predict(X)
            return y_true, y_pred        


    def get_trainable_parameters(self):
        if not hasattr(self.clf, "coef_"):
            raise ValueError("Model has not been fitted yet. Fit the model before counting parameters.")

        n_params = self.clf.coef_.size
        
        if hasattr(self.clf, "intercept_"):
            n_params += self.clf.intercept_.size

        return n_params



