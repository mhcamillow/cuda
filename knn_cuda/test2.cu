#include <iostream>
#include "double.cu"

using namespace std;

__global__
void teste_cuda (int *train_labels) {
    for (int i = 0; i < 10; i++) {
        if (i >= 5) {
            break;
        }
        for (int j = 0; j < 3; j++) {
            train_labels[i * 3 + j] = doubleIt(train_labels[i * 3 + j]);
        }
    }
}

int main(int argc, char *argv[]) {
    int * train_labels;
    cudaMallocManaged(&train_labels, 10 * 3 * sizeof(int));

    for (int i = 0; i < 30; i++) {
        train_labels[i] = i;
    }

    teste_cuda<<<1, 1>>>(&train_labels[0]);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            cout << train_labels[i * 3 + j] << endl;
        }
    }

    cudaFree(train_labels);
}