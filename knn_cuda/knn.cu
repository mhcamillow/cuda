#include <iostream>

using namespace std;

void cuda_guess(
    float * test_features,
    float * train_features,
    int * test_guesses, 
    int * train_labels,
    int * temp_irmao,
    int test_size, 
    int train_size, 
    int feature_count, 
    int k) 
{
    for (int testIdx = 0; testIdx < test_size; testIdx++) {
        float smallest_distance = 10.0f;
        int smallest_distance_id = 0;

        for (int trainIdx = 0; trainIdx < train_size; trainIdx++) {
            
            float distance = 0.0f;
            for (int featureIdx = 0; featureIdx < feature_count; featureIdx++) {
                distance = distance + powf(test_features[testIdx * feature_count + featureIdx] - train_features[trainIdx * feature_count + featureIdx], 2);
            }

            distance = sqrtf(distance);
            if (distance < smallest_distance) {
                smallest_distance = distance;
                smallest_distance_id = trainIdx;
            }
        }

        test_guesses[testIdx] = train_labels[smallest_distance_id];
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
            cuda_guess(
                test_features,
                train_features,
                test_guesses, 
                train_labels,
                temp_irmao,
                test_size, 
                train_size, 
                test_feature_count, 
                k
            );

            cudaDeviceSynchronize();
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