#include <iostream>
#include <vector>

using namespace std;

vector<int> init_vector(int size) {
    vector<int> v;
    for (auto i = 0; i < size; ++i)
        v.push_back(i);
    return v;
}

int main(int argc, char *argv[]) {
    int len = 50000000;
    int sum = 0;

    vector<int> v = init_vector(len);

    cout << len << endl;
    for (auto val : v) {
        sum += val;
    }
    cout << sum << endl;

    return 0;
}
