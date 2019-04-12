#include <iostream>

using namespace std;

void checkClosest(
    float * distances,
    int * ids,
    int pos,
    int k,
    float distance,
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

int guessClosest(int * train_labels,  int * ids, int pos, int k) {
    int np = 0, nf = 0;
    for (int i = 0; i < k; i++) {
        int idx = ids[pos + i];
        if (train_labels[idx] == 0) {
            nf++;
        } else {
            np++;
        }
    }
    if (np > nf)
        return 1;
    return 0;
}

void cuda_guess(
    float * test_features,
    float * train_features,
    int * test_guesses, 
    int * train_labels,
    float * distances,
    int * ids,
    int test_size, 
    int train_size, 
    int feature_count, 
    int k) 
{
    for (int testIdx = 0; testIdx < test_size; testIdx++) {
        int curr_pos = testIdx * k;
        for (int trainIdx = 0; trainIdx < train_size; trainIdx++) {
            
            float distance = 0.0f;
            for (int featureIdx = 0; featureIdx < feature_count; featureIdx++) {
                distance = distance + powf(test_features[testIdx * feature_count + featureIdx] - train_features[trainIdx * feature_count + featureIdx], 2);
            }
            distance = sqrtf(distance);

            checkClosest(distances, ids, curr_pos, k, distance, trainIdx);
        }

        test_guesses[testIdx] = guessClosest(train_labels, ids, curr_pos, k);
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

        void guess(float * train_features, int * train_labels, float * test_features, int * test_guesses) {
            float * distances;
            int * ids;
            cudaMallocManaged(&distances, train_size * k * sizeof(float));
            cudaMallocManaged(&ids, train_size * k * sizeof(int));
            
            for (int i = 0; i < train_size * k; i++) {
                distances[i] = 10000;
            }

            cuda_guess(
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
            cudaFree(ids);
        }

        float score(int * test_labels, int * test_guesses) {
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

            return (float)(tp + tn) / (test_size);
        }

        void printConfusionMatrix() {
            cout << tn << "\t" << fp << endl;
            cout << fn << "\t" << tp << endl;
        }
};