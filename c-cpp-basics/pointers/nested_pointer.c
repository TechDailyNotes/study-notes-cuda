#include <stdio.h>

int main() {
    int arr1[] = {11, 12, 13, 14};
    int arr2[] = {21, 22, 23, 24};
    int *ptr1 = arr1, *ptr2 = arr2;
    int *matrix[] = {ptr1, ptr2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("*matrix[i]++ = %d\nl", *matrix[i]++);
        }
    }

    return 0;
}
