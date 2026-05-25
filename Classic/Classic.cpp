#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;

Matrix build_weight_matrix(const vector<double>& arr) {
    int n = arr.size();
    Matrix W(n, Vector(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j)
                W[i][j] = exp(-abs(arr[i] - arr[j]));
    return W;
}

Matrix build_laplacian(const Matrix& W) {
    int n = W.size();
    Matrix L(n, Vector(n, 0.0));
    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++) deg += W[i][j];
        L[i][i] = deg;
        for (int j = 0; j < n; j++)
            if (i != j)
                L[i][j] = -W[i][j];
    }
    return L;
}

Vector matvec(const Matrix& A, const Vector& x) {
    int n = A.size();
    Vector y(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    return y;
}

double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

double norm(const Vector& a) {
    return sqrt(dot(a, a));
}

void orthogonalize(Vector& v) {
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= v.size();
    for (double& x : v) x -= mean;
}

Vector fiedler_vector(const Matrix& L, int max_iter = 1000, double tol = 1e-8) {
    int n = L.size();
    Vector v(n);
    random_device rd; mt19937 gen(rd()); uniform_real_distribution<> dis(-1, 1);
    for (int i = 0; i < n; i++) v[i] = dis(gen);
    orthogonalize(v); 
    double prev_lambda = 0.0;
    for (int iter = 0; iter < max_iter; iter++) {
        Vector Lv = matvec(L, v);
        orthogonalize(Lv);
        double nrm = norm(Lv);
        for (int i = 0; i < n; i++) v[i] = Lv[i] / nrm;
        double lambda = dot(v, matvec(L, v));
        if (abs(lambda - prev_lambda) < tol) break;
        prev_lambda = lambda;
    }
    return v;
}

int main() {
    setlocale(LC_CTYPE, "RUS");
    int n;
    cout << "Введите количество элементов: ";
    cin >> n;
    vector<double> arr(n);
    cout << "Введите элементы массива через пробел: ";
    for (int i = 0; i < n; i++) cin >> arr[i];

    Matrix W = build_weight_matrix(arr);

    Matrix L = build_laplacian(W);

    Vector fiedler = fiedler_vector(L);

    double min_f = *min_element(fiedler.begin(), fiedler.end());
    double max_f = *max_element(fiedler.begin(), fiedler.end());
    int min_idx = min_element(fiedler.begin(), fiedler.end()) - fiedler.begin();
    int max_idx = max_element(fiedler.begin(), fiedler.end()) - fiedler.begin();

    if (arr[min_idx] > arr[max_idx]) {
        for (double& x : fiedler) x = -x;
    }
    vector<pair<double, double>> paired;
    for (int i = 0; i < n; i++)
        paired.push_back({ fiedler[i], arr[i] });
    sort(paired.begin(), paired.end());

    cout << "SpectralSort: ";
    for (auto& p : paired)
        cout << p.second << " ";
    cout << endl;

    vector<double> arr_std = arr;
    sort(arr_std.begin(), arr_std.end());
    cout << "StdSort:      ";
    for (double x : arr_std)
        cout << x << " ";
    cout << endl;

    return 0;
}