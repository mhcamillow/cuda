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

using namespace std::chrono; 
using namespace std;

int FileUtils::getNumberOfElements(char *filepath) {
    ifstream inFile(filepath); 
    return count(istreambuf_iterator<char>(inFile), istreambuf_iterator<char>(), '\n');
}

int FileUtils::getNumberOfFeatures(char *filepath) {
    ifstream inFile(filepath);
    string line;

    if (inFile.is_open())
    {
        getline (inFile, line);
        inFile.close();
    }
    
    return count(line.begin(), line.end(), ' ') - 1;
}

void FileUtils::loadFeatures(string line, double * features) {
    string delimiter = ":";
    size_t pos = 0;
    int index = 0;

    string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
        line.erase(0, pos + 1);
        pos = line.find(" ");
        token = line.substr(0, pos);
        stringstream(token) >> features[index];
        index = index + 1;
        line.erase(0, pos + delimiter.length());
    }
}

void FileUtils::loadFile(char *filepath, int count, int * labels, double * features, int feature_count) {
    string line;
    ifstream inputFile (filepath);
    int index = 0;

    if (inputFile.is_open())
    {
        while ( getline (inputFile, line) )
        {
            labels[index] = line[0] - 48;
            line.erase(0, 2);
            loadFeatures(line, &features[index * feature_count]);
            index = index + 1;
        }
        inputFile.close();
    }
}