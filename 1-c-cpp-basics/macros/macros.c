#include <stdio.h>

#define PI 3.14159
#define area(r) (PI * (r) * (r))

#ifndef radius
#define radius 15
#endif

#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 8
#endif

int main() {
    printf("area(%d) = %.2f\n", radius, area(radius));
    return 0;
}
