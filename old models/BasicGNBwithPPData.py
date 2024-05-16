import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def main():
    # Load in data
    df = pd.read_csv('train_dataset.csv')

    D2V_genres = np.load('train_doc2vec_features_genre.npy')
    print(D2V_genres.shape)
    # Remove nominal and categorical data
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

    # Split training data up
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Standardise data
    standardiser = StandardScaler()
    standardised_X = standardiser.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(standardised_X, y, test_size=0.2)

    gnb = GaussianNB()

    k_folds = KFold(n_splits=10, shuffle=True)
    cv_scores = cross_val_score(gnb, X_train, y_train, cv=k_folds)

    print("Accuracy:", cv_scores)
    print("Mean Accuracy:", cv_scores.mean())




if __name__ == "__main__":
    main()