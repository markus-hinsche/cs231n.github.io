import numpy as np

from k_nearest_neighbor import KNearestNeighbor




def test_knn():

    # n_samples = 5000
    # n_features = 3000
    # n_samples_test = 500

    # n_samples = 300
    # n_features = 600
    # n_samples_test = 100

    n_samples = 3
    n_features = 4
    n_samples_test = 2

    a = np.arange(n_samples * n_features, dtype=np.float)
    X_train = a.reshape((n_samples, n_features ))
    y_train = np.arange(n_samples)

    clf = KNearestNeighbor()
    clf.train(X_train, y_train)

    a = np.arange(n_samples_test * n_features)
    X_test = a.reshape((n_samples_test , n_features))

    # expected = np.array([[  0. ,  8.,  16.],[  8.,   0.,   8.]])

    # dists_two = clf.compute_distances_two_loops(X_test)
    # assert np.all(dists_two == expected)

    # dists_one = clf.compute_distances_one_loop(X_test)
    # assert np.all(dists_two == dists_one)

    dists_no = clf.compute_distances_no_loops(X_test)
    # assert np.all(dists_one == dists_no)

    pred = clf.predict_labels(dists_no, k=2)
    print(pred)

if __name__ == '__main__':
    test_knn()
