#include <iostream>
#include <math.h>
#include <fstream>
#include <string.h>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <chrono> 
#include <cstdlib>
#include "fileUtils.hpp"
#include "knn.cu"

using namespace std::chrono; 
using namespace std;

char * train, * test;
int train_size, test_size, feature_count, k, * train_labels, * test_labels, * test_guesses, * closest_ids;
double * train_features, * test_features, * closest_distances;

void initialize(int argc, char *argv[]) {
    train = argv[1];
    test = argv[2];
    k = atoi(argv[3]);
    train_size = FileUtils::getNumberOfElements(train);
    test_size = FileUtils::getNumberOfElements(test);
    feature_count = FileUtils::getNumberOfFeatures(train);

    cudaMallocManaged(&train_labels, train_size * sizeof(int));
    cudaMallocManaged(&train_features, train_size * feature_count * sizeof(double));
    cudaMallocManaged(&test_labels, test_size * sizeof(int));
    cudaMallocManaged(&test_guesses, test_size * sizeof(int));
    cudaMallocManaged(&test_features, test_size * feature_count * sizeof(double));
    cudaMallocManaged(&closest_ids, test_size * k * sizeof(int));
    cudaMallocManaged(&closest_distances, test_size * k * sizeof(double));
}

void freeMemory() {
    cudaFree(train_labels);
    cudaFree(train_features);
    cudaFree(test_labels);
    cudaFree(test_guesses);
    cudaFree(test_features);
    cudaFree(closest_ids);
    cudaFree(closest_distances);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        cout << "knn <train> <test> <k>\n";
        return -1;
    }

    initialize(argc, argv);
    KNN knn(train_size, test_size, feature_count, k);

    FileUtils::loadFile(train, train_size, train_labels, train_features, feature_count);
    FileUtils::loadFile(test, test_size, test_labels, test_features, feature_count);

    auto start = std::chrono::high_resolution_clock::now();
    knn.train(train_features, train_labels);
    knn.guess(closest_distances, closest_ids, test_features, test_guesses);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time: " << duration.count() / 1000 << "ms" <<  endl;
    cout << "Score: " << knn.score(test_labels, test_guesses)  << endl;
    knn.printConfusionMatrix();

    freeMemory();

	return 0;
}