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
double * train_features, * test_features;

void initialize(int argc, char *argv[]) {
    filepath_train = argv[1];
    filepath_test = argv[2];
    k = atoi(argv[3]);
    train_size = FileUtils::getNumberOfElements(filepath_train);
    test_size = FileUtils::getNumberOfElements(filepath_test);
    feature_count = FileUtils::getNumberOfFeatures(filepath_train);

    cudaMallocManaged(&train_labels, train_size * sizeof(int));
    cudaMallocManaged(&train_features, train_size * feature_count * sizeof(double));
    cudaMallocManaged(&test_guesses, test_size * sizeof(int));
    cudaMallocManaged(&test_features, test_size * feature_count * sizeof(double));
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        cout << "knn <filepath_train> <filepath_test> <k>\n";
        return -1;
    }

    initialize(argc, argv);
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

    cout << "Free tudo-" << test_size <<  endl;

    cudaMallocManaged(&test_labels, test_size * sizeof(int));
    FileUtils::loadLabels(filepath_test, test_labels);

    cout << "Score: " << knn.score(test_labels, test_guesses) << "ms" <<  endl;
    cout << "Deveria: " << test_labels[0] <<  endl;
    cout << "Aqui já nao." <<  endl;
    
    knn.printConfusionMatrix();

    cudaFree(test_labels);
    cudaFree(test_guesses);

	return 0;
}