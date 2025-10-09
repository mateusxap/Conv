#include <immintrin.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <cmath>

static inline void pin_to_cpu0()
{
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

void avx_add(const float *__restrict a, const float *__restrict b, float *__restrict r, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 mul = _mm256_mul_ps(va, vb);
        __m256 div1 = _mm256_div_ps(va, vb);
        __m256 div2 = _mm256_div_ps(vb, va);
        __m256 add1 = _mm256_add_ps(mul, div1);
        __m256 add2 = _mm256_add_ps(div2, vb);
        __m256 add3 = _mm256_add_ps(add1, add2);
        __m256 add4 = _mm256_add_ps(add3, va);
        _mm256_storeu_ps(r + i, add4);
    }

    for (; i < n; ++i)
    {
        r[i] = a[i] * b[i] + a[i] / b[i] + b[i] / a[i] + b[i] + a[i];
    }
}

//__attribute__((optimize("no-tree-vectorize")))
void scalar_add(const float *a, const float *b, float *r, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        r[i] = a[i] * b[i] + a[i] / b[i] + b[i] / a[i] + b[i] + a[i];
}

int main()
{
    pin_to_cpu0();

    const size_t n = 200'000'000; // ~0.8 ГБ на массив
    std::vector<float> a(n, 1.0f), b(n, 2.0f), r(n, 0.0f);

    // прогрев
    avx_add(a.data(), b.data(), r.data(), n);
    scalar_add(a.data(), b.data(), r.data(), n);

    auto bench = [&](auto &&f, const char *name)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        f(a.data(), b.data(), r.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        // защита от DCE
        volatile float chk = r[0] + r[n / 2] + r[n - 1];
        std::cout << name << " ms: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << "  chk=" << chk << "\n";
    };

    bench(scalar_add, "scalar");
    bench(avx_add, "avx");
}
