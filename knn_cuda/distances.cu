#include <math.h>

__device__
double getEuclideanDistance(double * arr1, double * arr2, int features)
{
    double sum = 0;
    for (int i = 0; i < features; i++) {
        sum = sum + pow(arr1[i] - arr2[i], 2);
    }

	return sqrt(sum);
}