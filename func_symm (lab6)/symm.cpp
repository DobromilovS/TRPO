#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <omp.h>
#include <cblas.h> 

template <typename T>
void my_symm(char uplo, int m, int n, T alpha, const T* A, int ldA, const T* B, int ldB, T beta, T* C, int ldC) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = 0.0;
            for (int k = 0; k < m; ++k) {
                T a_val;
                if (uplo == 'U' || uplo == 'u') {
                    a_val = (i <= k) ? A[i * ldA + k] : A[k * ldA + i];
                } else {
                    a_val = (i >= k) ? A[i * ldA + k] : A[k * ldA + i];
                }
                sum += a_val * B[k * ldB + j];
            }
            int c_index = i * ldC + j;
            if (beta == 0.0) C[c_index] = alpha * sum;
            else C[c_index] = alpha * sum + beta * C[c_index];
        }
    }
}

template <typename T>
void fill_random(std::vector<T>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.1, 1.0);
    for (auto& val : vec) val = dis(gen);
}

double get_geo_mean(const std::vector<double>& times) {
    double log_sum = 0;
    for (double t : times) log_sum += std::log(t);
    return std::exp(log_sum / times.size());
}

int main() {
    const int m = 3000; 
    const int n = 3000;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<float> A(m * m), B(m * n), C_my(m * n), C_blas(m * n);
    fill_random(A);
    fill_random(B);

    int thread_variants[] = {1, 2, 4, 8, 16};

    std::cout << "Benchmarking Symm: Matrix " << m << "x" << n << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    for (int threads : thread_variants) {
        std::cout << "\n>>> TESTING WITH " << threads << " THREADS" << std::endl;


        std::vector<double> my_times;
        omp_set_num_threads(threads);
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            my_symm<float>('U', m, n, alpha, A.data(), m, B.data(), n, beta, C_my.data(), n);
            auto end = std::chrono::high_resolution_clock::now();
            my_times.push_back(std::chrono::duration<double>(end - start).count());
        }
        double my_mean = get_geo_mean(my_times);

        std::vector<double> blas_times;
        openblas_set_num_threads(threads);
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, alpha, A.data(), m, B.data(), n, beta, C_blas.data(), n);
            auto end = std::chrono::high_resolution_clock::now();
            blas_times.push_back(std::chrono::duration<double>(end - start).count());
        }
        double blas_mean = get_geo_mean(blas_times);

        std::cout << "My Implementation (GeoMean): " << std::fixed << std::setprecision(4) << my_mean << " sec" << std::endl;
        std::cout << "OpenBLAS (GeoMean):          " << blas_mean << " sec" << std::endl;
        std::cout << "Performance Ratio:           " << (blas_mean / my_mean) * 100.0 << "%" << std::endl;
    }

    return 0;
}

//g++ -O3 -march=native -fopenmp -o symm_benchmark symm_benchmark.cpp -lopenblas