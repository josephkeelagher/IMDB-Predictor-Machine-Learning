import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv('train_dataset.csv')
    print(df.head())

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    NBclassifier = GaussianNB()

    NBclassifier.fit(X_train, y_train)

    y_pred = NBclassifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)



if __name__ == "__main__":
    main()