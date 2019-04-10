#include <iostream>
using namespace std;

// int ** initializeArray(int size_i, int size_j) {
//     int ** arr;
//     arr = new int * [size_i];
//     for(int i = 0; i < size_i; i++) {
//         arr[i] = new int[size_j];
//     }
//     return arr;
// }

int main(int argc, char *argv[])
{
    int ** arr = initializeArray(10, 10);
    // arr = new int * [10];
    // for (int i = 0; i < 10; i++) {
    //     arr[i] = new int[10];
    // }
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            arr[i][j] = i + j;
        }
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << arr[i][j] << "\t";
        }
        cout << " \n";
    }
    cout << "\n";

 	return 0;
}