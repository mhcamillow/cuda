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

int guessClosest(int * train_labels,  float * distances, int * ids, int k, int classes) {
    int * guesses;
    int * guessDistances;
    cudaMallocManaged(&guesses, classes * sizeof(int));
    cudaMallocManaged(&guessDistances, classes * sizeof(float));

    for (int i = 0; i < classes; i++) {
        guesses[i] = 0;
        guessDistances[i] = 0;
    }
    
    for (int i = 0; i < k; i++) {
        int closestID = ids[i];
        int closestIDsLabel = train_labels[closestID];
        guesses[closestIDsLabel] = guesses[closestIDsLabel] + 1;
        guessDistances[closestIDsLabel] = guessDistances[closestIDsLabel] + distances[i];
    }

    int biggestCount = 0, biggestClass = -1, biggestClassDistance = 0;;
    for (int i = 0; i < classes; i++) {
        if ((guesses[i] > biggestCount) || (guesses[i] == biggestCount && guessDistances[i] < biggestClassDistance)) {
            biggestCount = guesses[i];
            biggestClass = i;
            biggestClassDistance = guessDistances[i];
        }
    }

    cout << endl;
    cudaFree(guesses);
    cudaFree(guessDistances);
    return biggestClass;
}

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
        int train_size, test_size, train_feature_count, test_feature_count, k, classes;
        int * confusionMatrix;
        KNN (int train_size_p, int test_size_p, int feature_count_p, int k_p, int classes_p) {
            train_size = train_size_p;
            test_size = test_size_p;
            train_feature_count = feature_count_p;
            test_feature_count = feature_count_p;
            k = k_p;
            classes = classes_p;
        }

        void guess(float * train_features, int * train_labels, float * test_features, int * test_guesses) {
            float * distances;
            float * kClosestDistances;
            int * kClosestIds;
            cudaMallocManaged(&distances, train_size * sizeof(float));
            cudaMallocManaged(&kClosestDistances, k * sizeof(float));
            cudaMallocManaged(&kClosestIds, k * sizeof(int));

            int blockSize = 256;
            int numBlocks = (train_size + blockSize - 1) / blockSize;

            for (int i = 0; i < test_size; i++) {
                calculateDistances<<<numBlocks, blockSize>>>(
                    &test_features[i * train_feature_count], 
                    train_features, 
                    distances, 
                    train_feature_count, 
                    train_size);
                cudaDeviceSynchronize();

                for (int j = 0; j < k; j++) {
                    kClosestDistances[j] = 1000;
                    kClosestIds[j] = -1;
                }

                for (int j = 0; j < train_size; j++) { 
                    // cout << "Distancia entre " << i << " e " << j << ": " << distances[j] << endl;
                    checkClosest(kClosestDistances, kClosestIds, k, distances[j], j);
                }

                for (int j = 0; j < k; j++) {
                    cout << "closests: " << kClosestIds[j] << ", ";
                }
                cout << endl;

                test_guesses[i] = guessClosest(train_labels, kClosestDistances, kClosestIds, k, classes);
            }

            cudaFree(distances);
            cudaFree(kClosestDistances);
            cudaFree(kClosestIds);
        }

        float score(int * test_labels, int * test_guesses) {
            cudaMallocManaged(&confusionMatrix, classes * classes * sizeof(int));
            int correctGuesses = 0;

            for (int i = 0; i < classes * classes; i++) { 
                confusionMatrix[i] = 0;
            }

            for (int i = 0; i < test_size; i++) {
                int guess = test_guesses[i];
                int actual = test_labels[i];

                // cout << "Sample " << i << " - " << guess << " - " << actual << endl;

                confusionMatrix[actual * classes + guess]++;
                if (guess == actual) { 
                    correctGuesses++;
                }
            }

            return (float)(correctGuesses) / (test_size);
        }

        void printConfusionMatrix() {
            for (int i = 0; i < classes * classes; i++) { 
                cout << confusionMatrix[i] << "\t";

                if ((i + 1) % (classes) == 0) {
                    cout << endl;
                }
            }
        }
};