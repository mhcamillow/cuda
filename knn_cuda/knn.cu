#include "distances.cu"
#include <iostream>

using namespace std;

__device__
void checkClosest(double distance, double * closest_distances, int * closest_ids, int train_idx) {
    if (distance < closest_distances[0]) {
        closest_distances[2] = closest_distances[1];
        closest_distances[1] = closest_distances[0];
        closest_distances[0] = distance;
        closest_ids[2] = closest_ids[1];
        closest_ids[1] = closest_ids[0];
        closest_ids[0] = train_idx;
    } else if (distance < closest_distances[1]) {
        closest_distances[2] = closest_distances[1];
        closest_distances[1] = distance;
        closest_ids[2] = closest_ids[1];
        closest_ids[1] = train_idx;
    } else if (distance < closest_distances[2]) {
        closest_distances[2] = distance;
        closest_ids[2] = train_idx;
    }
}

__device__
int check(int * closest_ids, int * train_labels, int k) {
    int n_count = 0;
    int p_count = 0;
    for (int j = 0; j < k; j++) {
        int id = closest_ids[j];
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

__global__
void cuda_guess( 
    double * closest_distances, 
    int * closest_ids, 
    double * test_features,
    double * train_features,
    int * test_guesses, 
    int * train_labels,
    int test_size, 
    int train_size, 
    int test_feature_count, 
    int k) 
{
    for (int i = 0; i < test_size; i++) {
        closest_distances[i * k + 0] = 1000;
        closest_distances[i * k + 1] = 1000;
        closest_distances[i * k + 2] = 1000;

        for (int j = 0; j < train_size; j++) {
            double distance = getEuclideanDistance(
                &test_features[i * test_feature_count], 
                &train_features[j * test_feature_count], 
                test_feature_count);
            checkClosest(distance, &closest_distances[i * k], &closest_ids[i * k], j);
        }

        test_guesses[i] = check(&closest_ids[i * k], train_labels, k);
    }
}

class KNN {
    public:
        int train_size, test_size, train_feature_count, test_feature_count, k, * train_labels;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        double * train_features;
        KNN (int train_size_p, int test_size_p, int feature_count_p, int k_p) {
            train_size = train_size_p;
            test_size = test_size_p;
            train_feature_count = feature_count_p;
            test_feature_count = feature_count_p;
            k = k_p;
        }

        void train(double * train_features_p, int * train_labels_p) {
            cout << "Training" << endl;
            train_labels = train_labels_p;
            train_features = train_features_p;
        }

        void guess(double * closest_distances, int * closest_ids, double * test_features, int * test_guesses) {
            cuda_guess<<<1, 1>>>(
                closest_distances, 
                closest_ids, 
                test_features,
                train_features,
                test_guesses, 
                train_labels,
                test_size, 
                train_size, 
                test_feature_count, 
                k
            );
            cudaDeviceSynchronize();
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
            cout << tn << "\t" << fp << endl;
            cout << fn << "\t" << tp << endl;
        }
};