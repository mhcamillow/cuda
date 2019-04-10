#include "distances.hpp"
#include <iostream>

class KNN {
    private: 
        void checkClosest(double distance, double ** closest_distances, int ** closest_ids, int element, int train_idx) {
            if (distance < closest_distances[element][0]) {
                closest_distances[element][2] = closest_distances[element][1];
                closest_distances[element][1] = closest_distances[element][0];
                closest_distances[element][0] = distance;
                closest_ids[element][2] = closest_ids[element][1];
                closest_ids[element][1] = closest_ids[element][0];
                closest_ids[element][0] = train_idx;
            } else if (distance < closest_distances[element][1]) {
                closest_distances[element][2] = closest_distances[element][1];
                closest_distances[element][1] = distance;
                closest_ids[element][2] = closest_ids[element][1];
                closest_ids[element][1] = train_idx;
            } else if (distance < closest_distances[element][2]) {
                closest_distances[element][2] = distance;
                closest_ids[element][2] = train_idx;
            }
        }

        int guess(int ** closest_ids, int * train_labels, int element_idx) {
            int n_count = 0;
            int p_count = 0;
            for (int j = 0; j < k; j++) {
                int id = closest_ids[element_idx][j];
                if (train_labels[id] == 0) {
                    n_count = n_count + 1;
                } else {
                    p_count = p_count + 1;
                }
            }

            if (p_count > n_count)
                return 1;

            return 0;
        }

    public:
        int train_size, test_size, train_feature_count, test_feature_count, k, * train_labels;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        double ** train_features;
        KNN (int train_size, int test_size, int feature_count, int k) {
            KNN::train_size = train_size;
            KNN::test_size = test_size;
            KNN::train_feature_count = feature_count;
            KNN::test_feature_count = feature_count;
            KNN::k = k;
        }

        void train(double ** train_features, int * train_labels) {
            KNN::train_labels = train_labels;
            KNN::train_features = train_features;
        }

        void guess(double ** closest_distances, int ** closest_ids, double ** test_features, int * test_guesses) {
            for (int i = 0; i < test_size; i++) {
                closest_distances[i][0] = 1000;
                closest_distances[i][1] = 1000;
                closest_distances[i][2] = 1000;

                for (int j = 0; j < train_size; j++) {
                    double distance = Distances::getEuclideanDistance(test_features[i], train_features[j], train_feature_count);
                    checkClosest(distance, closest_distances, closest_ids, i, j);
                }

                test_guesses[i] = guess(closest_ids, train_labels, i);
            }
        }

        double score(int * test_labels, int * test_guesses) {
            for (int i = 0; i < test_size; i++) {
                if (test_labels[i] == test_guesses[i]) {
                    if (test_labels[i] == 0) {
                        tn++;
                    } else {
                        tp++;
                    }
                } else { 
                    if (test_labels[i] == 0) {
                        fp++;
                    } else {
                        fn++;
                    }
                }
            }

            return (double)(tp + tn) / (test_size);
        }

        void printConfusionMatrix() {
            std::cout << tn << "\t" << fp << std::endl;
            std::cout << fn << "\t" << tp << std::endl;
        }
};