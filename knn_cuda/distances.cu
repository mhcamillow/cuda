#include <math.h>

__device__
double getEuclideanDistance(double * arr1, double * arr2, int i, int j, int features)
{
    double sum = 0;
    for (int x = 0; x < features; x++) {
        sum = sum + pow(arr1[i * features + x] - arr2[j * features + x], 2);
    }

	return sqrt(sum);
}