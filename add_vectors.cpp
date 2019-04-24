#include <iostream>
#include <vector>
#include <ctime>

using namespace std;

vector<int> init_vector(int size) {
//void init_vector(int size, vector<int> &v) {
    vector<int> v;
    for (auto i = 0; i < size; ++i)
        v.push_back(i);
    return v;
}

void add(vector<int> &a, const vector<int> &b, vector<int> &c) {
    int i = 0;
    for (vector<int>::iterator it = a.begin(); it != a.end(); ++it, ++i) {
        c.push_back(*it + b[i]);
    }
}

int main(int argc, char *argv[]) {
    //int len = 500000000;
    //const int len = 500000000;
    const int len = 10000000;
    time_t old_time, new_time;

    vector<int> a = init_vector(len);
    vector<int> b = init_vector(len);
 //   vector<int> a;
 //   vector<int> b;
    vector<int> c;
 //   init_vector(len, a);
 //   init_vector(len, b);

    time(&old_time);
    add(a, b, c);
    time(&new_time);

    cout << c.size() << "  " << new_time - old_time << endl;

    return 0;
}
