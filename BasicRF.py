import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def main():
    # Load in data
    df = pd.read_csv('train_dataset.csv')
    test_df = pd.read_csv('test_dataset.csv')

    # Remove nominal and categorical data
    id_col = test_df['id']
    df.drop('id',axis=1,inplace=True)
    df.drop('director_name', axis=1, inplace=True)
    df.drop('actor_2_name', axis=1, inplace=True)
    df.drop('genres', axis=1, inplace=True)
    df.drop('actor_1_name', axis=1, inplace=True)
    df.drop('movie_title', axis=1, inplace=True)
    df.drop('actor_3_name', axis=1, inplace=True)
    df.drop('plot_keywords', axis=1, inplace=True)
    df.drop('title_embedding', axis=1, inplace=True) #FastText vectorized, not sure how it works yet
    df.drop('language', axis=1, inplace=True)
    df.drop('country', axis=1, inplace=True)
    df.drop('content_rating', axis=1, inplace=True)

    test_df.drop('id',axis=1,inplace=True)
    test_df.drop('director_name', axis=1, inplace=True)
    test_df.drop('actor_2_name', axis=1, inplace=True)
    test_df.drop('genres', axis=1, inplace=True)
    test_df.drop('actor_1_name', axis=1, inplace=True)
    test_df.drop('movie_title', axis=1, inplace=True)
    test_df.drop('actor_3_name', axis=1, inplace=True)
    test_df.drop('plot_keywords', axis=1, inplace=True)
    test_df.drop('title_embedding', axis=1, inplace=True) #FastText vectorized, not sure how it works yet
    test_df.drop('language', axis=1, inplace=True)
    test_df.drop('country', axis=1, inplace=True)
    test_df.drop('content_rating', axis=1, inplace=True)

    

    # Split training data up
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Standardise data
    standardiser = StandardScaler()
    standardised_X = standardiser.fit_transform(X)

    # Split training data into test and train partitions
    X_train, X_test, y_train, y_test = train_test_split(standardised_X, y, test_size=0.2)

    # Initiate model and use k-folds to get a mean accuracy of 10-folds
    rf = RandomForestClassifier()

    k_folds = KFold(n_splits=10, shuffle=True)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=k_folds)

    # Fit the model to training data and test it on the 20% as test data
    rf.fit(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)

    print("K-Folds Mean Accuracy:", cv_scores.mean())
    print("Test Split Accuracy:", test_accuracy)

    # Make test set predictions for Kaggle
    predictions = rf.predict(test_df)
    df_predictions = pd.DataFrame({'id': id_col, 'imdb_score_binned': predictions})
    df_predictions.to_csv('predictions.csv', index=False)






if __name__ == "__main__":
    main()