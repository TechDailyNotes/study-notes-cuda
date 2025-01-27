#include <iostream>

using namespace std;

int main() {
    // Correct digit separator.
    const int N = 10'000'000;
    cout << "N = " << N << endl;

    // Incorrect digit separator.
    // const int M = 10_000_000;
    // cout << "M = " << M << endl;

    return 0;
}
