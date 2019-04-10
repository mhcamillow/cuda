#include <iostream>
#include <math.h>
#include <fstream>
#include <string.h>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <chrono> 
#include <cstdlib>
#include "util.hpp"
#include "fileUtils.hpp"
#include "knn.cpp"

using namespace std::chrono; 
using namespace std;

char * train, * test;
int train_size, test_size, feature_count, k, * train_labels, * test_labels, * test_guesses, ** closest_ids;
double ** train_features, ** test_features, ** closest_distances;

void initialize(int argc, char *argv[]) {
    train = argv[1];
    test = argv[2];
    k = atoi(argv[3]);
    train_size = FileUtils::getNumberOfElements(train);
    test_size = FileUtils::getNumberOfElements(test);
    feature_count = FileUtils::getNumberOfFeatures(train);

    train_labels = new int [train_size];
    train_features = util::initializeArrayDouble(train_size, feature_count);
    cout << "Loading train file" << "\n";

    test_labels = new int [train_size];
    test_guesses = new int [train_size];
    test_features = util::initializeArrayDouble(train_size, feature_count);
    cout << "Loading test file" << "\n";

    closest_ids = util::initializeArrayInt(train_size, k);
    closest_distances = util::initializeArrayDouble(train_size, k);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        cout << "knn <train> <test> <k>\n";
        return -1;
    }

    initialize(argc, argv);
    KNN knn(train_size, test_size, feature_count, k);

    FileUtils::loadFile(train, train_size, train_labels, train_features);
    FileUtils::loadFile(test, train_size, test_labels, test_features);

    auto start = std::chrono::high_resolution_clock::now();
    knn.train(train_features, train_labels);
    knn.guess(closest_distances, closest_ids, test_features, test_guesses);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time: " << duration.count() / 1000 << "ms" <<  endl;
    cout << "Score: " << knn.score(test_labels, test_guesses)  << endl;
    knn.printConfusionMatrix();

	return 0;
}