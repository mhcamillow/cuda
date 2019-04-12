#include <math.h>

__device__
float getEuclideanDistance(float * arr1, float * arr2, int i, int j, int features)
{
    float sum = 0;
    for (int x = 0; x < features; x++) {
        sum = sum + pow(arr1[i * features + x] - arr2[j * features + x], 2);
    }

	return sqrt(sum);
}