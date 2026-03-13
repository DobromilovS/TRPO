#include <iostream>

template <typename T>
void my_symm(
    char uplo, 
    int m, int n, 
    T alpha, 
    const T* A, int ldA, 
    const T* B, int ldB, 
    T beta, 
    T* C, int ldC)
    {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = 0.0;

                for (int k = 0; k < m; ++k) {
                    T a_val; 
                    if (uplo == 'U' || uplo == 'u') {
                        if (i <= k) {
                            a_val = A[i * ldA + k];
                        } else {
                            a_val = A[k * ldA + i]; 
                        }
                    } 
                    T b_val = B[k * ldB + j];
                    sum += a_val * b_val;
                }

                int c_index = i * ldC + j;
                
                if (beta == 0.0) {
                    C[c_index] = alpha * sum;
                } else {
                    C[c_index] = alpha * sum + beta * C[c_index];
                }
            }
        }
    }


int main(){
    int m=2;
    int n=2;

    float A[] = {
        1.0f, 2.0f,
        0.0f, 3.0f
    };

    float B[] = {
        1.0f, 1.0f,
        1.0f, 1.0f
    };

    float C[] = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };

    float alpha = 1.0f;
    float beta = 0.0f;

    //C=a*(A*B)+b*C

    my_symm<float>('U', m, n, alpha, A, m, B, n, beta, C, n);
    for(int i=0; i<m; ++i) {
        for(int j=0; j<n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}