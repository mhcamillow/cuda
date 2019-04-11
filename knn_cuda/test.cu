#include <iostream>
#include "double.cu"

using namespace std;

__device__
void checkClosest(
    double * distances,
    int * ids,
    int pos,
    int k,
    double distance,
    int trainIdx
) 
{
    for (int i = 0; i < k; i++) {
        if (distance < distances[pos + i]) {
            for (int x = k - 1; x > i; x--) {
                distances[pos + x] = distances[pos + x - 1];
                ids[pos + x] = ids[pos + x - 1];
            }
            distances[pos + i] = distance;
            ids[pos + i] = trainIdx;
            return;
        }
    }
}

__global__
void doStuff(double * distances, int * ids, int test_size, int k){
    double d1 = 0.345;
    double d2 = 0.245;
    double d3 = 0.145;
    double d4 = 0.545;
    double d5 = 0.645;
    double d6 = 0.445;
    double d7 = 0.045;
    double d8 = 0.945;
    int pos = (test_size - 1) * k;
    checkClosest(distances, ids, pos, k, d1, 1); 
    checkClosest(distances, ids, pos, k, d2, 2); 
    checkClosest(distances, ids, pos, k, d3, 3); 
    checkClosest(distances, ids, pos, k, d4, 4); 
    checkClosest(distances, ids, pos, k, d5, 5); 
    checkClosest(distances, ids, pos, k, d6, 6); 
    checkClosest(distances, ids, pos, k, d7, 7); 
    checkClosest(distances, ids, pos, k, d8, 8); 
}

__global__
void teste_cuda (int *train_labels, int ok) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 3; j++) {
            train_labels[i * 3 + j] = doubleIt(train_labels[i * 3 + j]) + ok;
        }
    }
}

int main(int argc, char *argv[]) {
    // int * train_labels;
    // cudaMallocManaged(&train_labels, 10 * 3 * sizeof(int));

    // for (int i = 0; i < 30; i++) {
    //     train_labels[i] = i;
    // }
    // int ok = 2;
    // teste_cuda<<<1, 1>>>(&train_labels[0], ok);
    // cudaDeviceSynchronize();

    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         cout << train_labels[i * 3 + j] << endl;
    //     }
    // }

    // cudaFree(train_labels);
    int test_size = 3;
    int k = 3;
    double * distances;
    int * ids;
    cudaMallocManaged(&distances, test_size * k * sizeof(double));
    cudaMallocManaged(&ids, test_size * k * sizeof(int));
    
    for (int i = 0; i < test_size * k; i++) {
        distances[i] = 1000;
        // ids[i] = -1;
    }

    doStuff<<<1,1>>>(distances, ids, test_size, k);
    cudaDeviceSynchronize();

    for (int i = 0; i < test_size * k; i++) {
        cout << "Idx: " << i << " - ID: " << ids[i] << " - Distance: " <<  (double)distances[i] << endl;
    }

    cudaFree(distances);
    cudaFree(ids);
}