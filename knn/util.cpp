#include "util.h"
using namespace std;

double ** util::initializeArray(int size_i, int size_j) {
    double ** array;
    array = new double *[size_i];
    for(int i = 0; i < size_i; i++) {
        array[i] = new double[size_j];
    }
    return array;
}