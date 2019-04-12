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

char * filepath_train, * filepath_test;
int train_size, test_size, feature_count, k, * train_labels, * test_labels, * test_guesses;
float * train_features, * test_features;

int main(int argc, char *argv[])
{
    cout << "=================== KNN K-BULOZO ========================" << endl;
    if (argc < 4) {
        cout << "knn <filepath_train> <filepath_test> <k>\n";
        return -1;
    }

    filepath_train = argv[1];
    filepath_test = argv[2];
    k = atoi(argv[3]);

    train_size = FileUtils::getNumberOfElements(filepath_train);
    test_size = FileUtils::getNumberOfElements(filepath_test);
    feature_count = FileUtils::getNumberOfFeatures(filepath_train);

    cout << "Train file: " << filepath_train << endl;
    cout << "Test file: " << filepath_test << endl;
    cout << "K Value: " << k << endl;
    cout << "Train size: " << train_size << endl;
    cout << "Test size: " << test_size << endl;
    cout << "Number of features: " << feature_count << endl;

    cudaMallocManaged(&train_labels, train_size * sizeof(int));
    cudaMallocManaged(&train_features, train_size * feature_count * sizeof(float));
    cudaMallocManaged(&test_guesses, test_size * sizeof(int));
    cudaMallocManaged(&test_features, test_size * feature_count * sizeof(float));

    KNN knn(train_size, test_size, feature_count, k);

    FileUtils::loadFeatures(filepath_train, train_features, feature_count);
    FileUtils::loadLabels(filepath_train, train_labels);

    auto start = std::chrono::high_resolution_clock::now();
    FileUtils::loadFeatures(filepath_test, test_features, feature_count);

    knn.guess(train_features, train_labels, test_features, test_guesses);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Total time: " << duration.count() / 1000 << "ms" <<  endl;

    cudaFree(train_labels);
    cudaFree(train_features);
    cudaFree(test_features);

    cudaMallocManaged(&test_labels, test_size * sizeof(int));
    FileUtils::loadLabels(filepath_test, test_labels);

    cout << "Score: " << knn.score(test_labels, test_guesses) <<  endl;

    knn.printConfusionMatrix();

    cudaFree(test_labels);
    cudaFree(test_guesses);

    cout << "=========================================================" << endl;

	return 0;
}