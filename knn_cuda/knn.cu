#include "distances.cu"
#include <iostream>

using namespace std;

__device__
void checkClosest(
    double * distances,
    int * ids,
    int pos,
    int k,
    double distance,
    int trainIdx
) 
{
    for (int i = 0; i < k; i++) {
        if (distance < distances[pos + i]) {
            for (int x = k - 1; x > i; x--) {
                distances[pos + x] = distances[pos + x - 1];
                ids[pos + x] = ids[pos + x - 1];
            }
            distances[pos + i] = distance;
            ids[pos + i] = trainIdx;
            return;
        }
    }
}

__device__
int check(int * closest_ids, int * train_labels, int i, int k) {
    int n_count = 0;
    int p_count = 0;
    // for (int j = 0; j < k; j++) {
    //     int id = closest_ids[i * k + j];
    //     if (train_labels[id] == 0) {
    //         n_count = n_count + 1;
    //     } else {
    //         p_count = p_count + 1;
    //     }
    // }

    if (p_count > n_count)
        return 1;

    return 0;
}

__global__
void cuda_guess(
    double * test_features,
    double * train_features,
    int * test_guesses, 
    int * train_labels,
    double * distances,
    int * ids,
    int test_size, 
    int train_size, 
    int test_feature_count, 
    int k) 
{
    for (int i = 0; i < test_size; i++) {
        int curr_pos = i * train_size;
        for (int x = 0; x < k; x++) {
            distances[curr_pos * x] = 1000;
            ids[curr_pos * x] = 0;
        }

        for (int j = 0; j < train_size; j++) {
            double distance = getEuclideanDistance(test_features, train_features, i, j, test_feature_count);
            checkClosest(distances, ids, curr_pos, k, distance, j);
        }

        test_guesses[i] = 1;//check(ids, train_labels, i, k);
    }
}

__global__
void cuda_test(int * train_labels, int size) {
    for (int i = 0; i < size; i++) {
        train_labels[i] = i;
    }
}

class KNN {
    public:
        int train_size, test_size, train_feature_count, test_feature_count, k;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        KNN (int train_size_p, int test_size_p, int feature_count_p, int k_p) {
            train_size = train_size_p;
            test_size = test_size_p;
            train_feature_count = feature_count_p;
            test_feature_count = feature_count_p;
            k = k_p;
        }

        void guess(double * train_features, int * train_labels, double * test_features, int * test_guesses) {
            double * distances;
            int * ids;
            cudaMallocManaged(&distances, test_size * k * sizeof(double));
            cudaMallocManaged(&ids, test_size * k * sizeof(int));

            cuda_guess<<<1, 1>>>(
                test_features,
                train_features,
                test_guesses, 
                train_labels,
                distances,
                ids,
                test_size, 
                train_size, 
                test_feature_count, 
                k
            );
            cudaDeviceSynchronize();
            cudaFree(distances);
        }

        void test(int * train_labels, int size) {
            cuda_test<<<1, 1>>>(train_labels, size);
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