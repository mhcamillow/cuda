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
#include "distances.hpp"
#include "fileUtils.hpp"

using namespace std::chrono; 
using namespace std;

int train_samples_count, test_samples_count, train_feature_count, test_feature_count, k;
char * train, * test;

void initialize() {
    train_samples_count = FileUtils::getNumberOfElements(train);
    cout << "Train elements: " << train_samples_count << "\n";
    train_feature_count = FileUtils::getNumberOfFeatures(train);
    cout << "Train features: " << train_feature_count << "\n";
    test_feature_count = FileUtils::getNumberOfFeatures(test);
    cout << "Test features: " << test_feature_count << "\n";

    cout << "K-Value: " << k << "\n";

    test_samples_count = FileUtils::getNumberOfElements(test);
}

void checkClosest(double distance, double ** closest_distances, int ** closest_ids, int element, int train_idx) {
    if (distance < closest_distances[element][0]) {
        closest_distances[element][2] = closest_distances[element][1];
        closest_distances[element][1] = closest_distances[element][0];
        closest_distances[element][0] = distance;
        closest_ids[element][2] = closest_ids[element][1];
        closest_ids[element][1] = closest_ids[element][0];
        closest_ids[element][0] = train_idx;
    } else if (distance < closest_distances[element][1]) {
        closest_distances[element][2] = closest_distances[element][1];
        closest_distances[element][1] = distance;
        closest_ids[element][2] = closest_ids[element][1];
        closest_ids[element][1] = train_idx;
    } else if (distance < closest_distances[element][2]) {
        closest_distances[element][2] = distance;
        closest_ids[element][2] = train_idx;
    }
}

int guess(int ** closest_ids, int * train_labels, int element_idx) {
    int n_count = 0;
    int p_count = 0;
    for (int j = 0; j < 3; j++) {
        int id = closest_ids[element_idx][j];
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

void fit(double ** closest_distances, int ** closest_ids, double ** test_features, double ** train_features, int * train_labels, int * test_guesses) {
    for (int i = 0; i < test_samples_count; i++) {
        closest_distances[i][0] = 1000;
        closest_distances[i][1] = 1000;
        closest_distances[i][2] = 1000;

        for (int j = 0; j < train_samples_count; j++) {
            double distance = Distances::getEuclideanDistance(test_features[i], train_features[j], train_feature_count);
            checkClosest(distance, closest_distances, closest_ids, i, j);
        }

        test_guesses[i] = guess(closest_ids, train_labels, i);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        cout << "knn <train> <test> <k>\n";
        return -1;
    }

    train = argv[1];
    test = argv[2];
    k = atoi(argv[3]);

    initialize();

    if (train_feature_count != test_feature_count) {
        cout << "train_feature_count != test_feature_count" << "\n";
        return -1;
    }

    int train_labels[train_samples_count];
    double ** train_features = util::initializeArrayDouble(train_samples_count, train_feature_count);
    cout << "Loading train file" << "\n";
    FileUtils::loadFile(train, train_samples_count, train_labels, train_features);

    int test_labels[test_samples_count];
    int test_guesses[test_samples_count];
    double ** test_features = util::initializeArrayDouble(test_samples_count, train_feature_count);
    cout << "Loading test file" << "\n";
    FileUtils::loadFile(test, test_samples_count, test_labels, test_features);

    int ** closest_ids = util::initializeArrayInt(test_samples_count, k);
    double ** closest_distances = util::initializeArrayDouble(test_samples_count, k);

    auto start = std::chrono::high_resolution_clock::now();
    fit(closest_distances, closest_ids, test_features, train_features, train_labels, test_guesses);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time: " << duration.count() / 1000 << "ms" <<  endl;

    // cout << "Time - Distance: " << time_distance / 1000 << "ms" <<  endl;
    // cout << "Time - Moving: " << time_moving / 1000 << "ms" <<  endl;

    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < test_samples_count; i++) {
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

    cout << "Score: " << (double)(tp + tn) / (test_samples_count) << endl;
    cout << tn << "\t" << fp << endl;
    cout << fn << "\t" << tp << endl;
   
	return 0;
}



// auto d_start = std::chrono::high_resolution_clock::now();
// double distance = Distances::getEuclideanDistance(test_features[i], train_features[j], train_feature_count);
// auto d_total = std::chrono::high_resolution_clock::now() - d_start;
// time_distance = time_distance + std::chrono::duration_cast<std::chrono::microseconds>(d_total).count();