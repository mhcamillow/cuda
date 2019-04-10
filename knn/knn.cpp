#include <iostream>
#include <math.h>
#include <fstream>
#include <string.h>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <chrono> 
#include <cstdlib>
#include "util.h"

using namespace std::chrono; 
using namespace std;

int getNumberOfElements(char *filepath) {
    ifstream inFile(filepath); 
    return count(istreambuf_iterator<char>(inFile), istreambuf_iterator<char>(), '\n');
}

int getNumberOfFeatures(char *filepath) {
    ifstream inFile(filepath);
    string line;

    if (inFile.is_open())
    {
        getline (inFile, line);
        inFile.close();
    }
    
    return count(line.begin(), line.end(), ' ') - 1;
}

void loadFeatures(string line, double * features) {
    string delimiter = ":";
    size_t pos = 0;
    int index = 0;

    string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
        line.erase(0, pos + 1);
        // cout << line << endl;
        pos = line.find(" ");
        token = line.substr(0, pos);
        stringstream(token) >> features[index];
        index = index + 1;
        line.erase(0, pos + delimiter.length());
    }
}

void loadFile(char *filepath, int count, int * labels, double ** features) {
    string line;
    ifstream inputFile (filepath);
    int index = 0;

    if (inputFile.is_open())
    {
        while ( getline (inputFile, line) )
        {
            labels[index] = line[0] - 48;
            line.erase(0, 2);
            loadFeatures(line, features[index]);
            index = index + 1;
        }
        inputFile.close();
    }
}

double getEuclideanDistance(double arr1[], double arr2[], int features)
{
    double sum = 0;
    for (int i = 0; i < features; i++) {
        sum = sum + pow(arr1[i] - arr2[i], 2);
    }

	return sqrt(sum);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        cout << "knn <train> <test> <k>\n";
        return -1;
    }

    char *train = argv[1];
    char *test = argv[2];
    int k = atoi(argv[3]);

    int train_samples_count = getNumberOfElements(train);
    cout << "Train elements: " << train_samples_count << "\n";
    int train_feature_count = getNumberOfFeatures(train);
    cout << "Train features: " << train_feature_count << "\n";
    int test_feature_count = getNumberOfFeatures(test);
    cout << "Test features: " << test_feature_count << "\n";

    cout << "K-Value: " << k << "\n";

    if (train_feature_count != test_feature_count) {
        cout << "train_feature_count != test_feature_count" << "\n";
        return -1;
    }
    
    int train_labels[train_samples_count];
    double ** train_features = util::initializeArray(train_samples_count, train_feature_count);
    cout << "Loading train file" << "\n";
    loadFile(train, train_samples_count, train_labels, train_features);


    int test_samples_count = getNumberOfElements(test);
    int test_labels[test_samples_count];
    int test_guesses[test_samples_count];
    double ** test_features = util::initializeArray(test_samples_count, train_feature_count);
    cout << "Loading test file" << "\n";
    loadFile(test, test_samples_count, test_labels, test_features);

    auto start = high_resolution_clock::now();
    long long time_distance = 0, time_moving = 0;
    int closest_ids[test_samples_count][k];
    double closest_distances[test_samples_count][k];
    for (int i = 0; i < test_samples_count; i++) {
        closest_distances[i][0] = 1000;
        closest_distances[i][1] = 1000;
        closest_distances[i][2] = 1000;

        for (int j = 0; j < train_samples_count; j++) {
            auto d_start = std::chrono::high_resolution_clock::now();
            double distance = getEuclideanDistance(test_features[i], train_features[j], train_feature_count);
            auto d_total = std::chrono::high_resolution_clock::now() - d_start;
            time_distance = time_distance + std::chrono::duration_cast<std::chrono::microseconds>(d_total).count();

            auto m_start = std::chrono::high_resolution_clock::now();
            if (distance < closest_distances[i][0]) {
                closest_distances[i][2] = closest_distances[i][1];
                closest_distances[i][1] = closest_distances[i][0];
                closest_distances[i][0] = distance;
                closest_ids[i][2] = closest_ids[i][1];
                closest_ids[i][1] = closest_ids[i][0];
                closest_ids[i][0] = j;
            } else if (distance < closest_distances[i][1]) {
                closest_distances[i][2] = closest_distances[i][1];
                closest_distances[i][1] = distance;
                closest_ids[i][2] = closest_ids[i][1];
                closest_ids[i][1] = j;
            } else if (distance < closest_distances[i][2]) {
                closest_distances[i][2] = distance;
                closest_ids[i][2] = j;
            }
            auto m_total = std::chrono::high_resolution_clock::now() - m_start;
            time_moving = time_moving + std::chrono::duration_cast<std::chrono::microseconds>(m_total).count();
        }

        int n_count = 0;
        int p_count = 0;
        for (int j = 0; j < 3; j++) {
            int id = closest_ids[i][j];
            if (train_labels[id] == 0) {
                n_count = n_count + 1;
            } else {
                p_count = p_count + 1;
            }
        }

        if (p_count > n_count) {
            test_guesses[i] = 1;
        } else {
            test_guesses[i] = 0;
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time: " << duration.count() / 1000 << "ms" <<  endl;
    cout << "Time - Distance: " << time_distance / 1000 << "ms" <<  endl;
    cout << "Time - Moving: " << time_moving / 1000 << "ms" <<  endl;

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