#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *ptr = NULL;
    printf("ptr = %p\n", (void*) ptr);  // ptr = (nil)

    printf("call malloc()\n");
    ptr = (int*) malloc(sizeof(int));
    assert(ptr != NULL);
    printf("ptr != NULL\n");
    printf("ptr = %p\n", (void*) ptr);  // ptr = 0x595ebcc3a6b0

    printf("call free()\n");
    free(ptr);
    ptr = NULL;
    assert(ptr == NULL);
    printf("ptr == NULL\n");
    printf("ptr = %p\n", (void*) ptr);  // ptr = (nil)
}
