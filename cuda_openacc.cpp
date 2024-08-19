#include <stdlib.h>

int main() {
  int n = 1 << 10;
  int *a = new int[n];
  int *b = new int[n];
  int *c = new int[n];

  srand(1);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = rand() % 100;
      b[i * n + j] = rand() % 100;
    }
  }

  #pragma acc kernels copyin(a[0:n*n], b[0:n*n]), copy(c[0:n*n])
  {
    #pragma acc loop independent
    for (int i = 0; i < n; i++) {
      #pragma acc loop independent
      for (int j = 0; j < n; j++) {
        int sum = 0;
        #pragma acc loop independent reduction(+:sum)
        for (int k = 0; k < n; k++) {
          sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
      }
    }
  }

  delete[] a;
  delete[] b;
  delete[] c;
  
  return 0;
}
