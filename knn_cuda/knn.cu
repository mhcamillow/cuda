#include <iostream>

using namespace std;

void checkClosest(
    float * distances,
    int * ids,
    int k,
    float distance,
    int trainIdx
) 
{
    for (int i = 0; i < k; i++) {
        if (distance < distances[i]) {
            for (int x = k - 1; x > i; x--) {
                distances[x] = distances[x - 1];
                ids[x] = ids[x - 1];
            }
            distances[i] = distance;
            ids[i] = trainIdx;
            return;
        }
    }
}

int guessClosest(int * train_labels,  int * ids, int k) {
    int np = 0, nf = 0;
    for (int i = 0; i < k; i++) {
        int idx = ids[i];
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

// void cuda_guess(
//     float * test_features,
//     float * train_features,
//     int * test_guesses, 
//     int * train_labels,
//     float * distances,
//     int * ids,
//     int test_size, 
//     int train_size, 
//     int feature_count, 
//     int k) 
// {
//     for (int testIdx = 0; testIdx < test_size; testIdx++) {
//         int curr_pos = testIdx * k;
//         for (int trainIdx = 0; trainIdx < train_size; trainIdx++) {
            
//             float distance = 0.0f;
//             for (int featureIdx = 0; featureIdx < feature_count; featureIdx++) {
//                 distance = distance + powf(test_features[testIdx * feature_count + featureIdx] - train_features[trainIdx * feature_count + featureIdx], 2);
//             }
//             distance = sqrtf(distance);

//             checkClosest(distances, ids, curr_pos, k, distance, trainIdx);
//         }

//         test_guesses[testIdx] = guessClosest(train_labels, ids, curr_pos, k);
//     }
// }

__global__
void calculateDistances(float * test_features, float * train_features, float * distances, int feature_count, int train_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < train_size; i += stride) {
        float distance = 0.0f;
        for (int featureIdx = 0; featureIdx < feature_count; featureIdx++) {
            distance = distance + powf(test_features[featureIdx] - train_features[i * feature_count + featureIdx], 2);
        }
    
        distances[i] = sqrtf(distance);
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
            float * kClosestDistances;
            int * kClosestIds;
            cudaMallocManaged(&distances, train_size * sizeof(float));
            cudaMallocManaged(&kClosestDistances, k * sizeof(float));
            cudaMallocManaged(&kClosestIds, k * sizeof(int));

            // for (int i = 0; i < test_size; i++) {
            //     distances[i] = 0.12345;
            // }

            int blockSize = 256;
            int numBlocks = (train_size + blockSize - 1) / blockSize;
            // <<<numBlocks, blockSize>>>

            for (int i = 0; i < test_size; i++) {
            // for (int i = 0; i < 50; i++) {
                // cout << "Runnning test " << i << endl;
                calculateDistances<<<numBlocks, blockSize>>>(
                    &test_features[i * train_feature_count], 
                    train_features, 
                    distances, 
                    train_feature_count, 
                    train_size);
                cudaDeviceSynchronize();

                // for (int j = 0; j < train_size; j++) { 
                //     cout << "Id: " << j << ": " << distances[j] << endl;
                // }

                for (int j = 0; j < k; j++) {
                    kClosestDistances[j] = 100;
                    kClosestIds[j] = -1;
                }

                for (int j = 0; j < train_size; j++) { 
                    checkClosest(kClosestDistances, kClosestIds, k, distances[j], j);
                }

                // cout << "For test " << i << ", closest ids: " << endl;
                // for ( int a = 0; a < k; a++) {
                //     cout << "Id: " << kClosestIds[a] << ": " << kClosestDistances[a] << " - ";
                // }
                // cout << endl;

                test_guesses[i] = guessClosest(train_labels, kClosestIds, k);
            }

            cudaFree(distances);
            cudaFree(kClosestDistances);
            cudaFree(kClosestIds);
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