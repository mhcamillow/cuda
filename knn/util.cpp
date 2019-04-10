#include "util.hpp"
using namespace std;

double ** util::initializeArrayDouble(int size_i, int size_j) {
    double ** array;
    array = new double *[size_i];
    for(int i = 0; i < size_i; i++) {
        array[i] = new double[size_j];
    }
    return array;
}

int ** util::initializeArrayInt(int size_i, int size_j) {
    int ** array;
    array = new int *[size_i];
    for(int i = 0; i < size_i; i++) {
        array[i] = new int[size_j];
    }
    return array;
}