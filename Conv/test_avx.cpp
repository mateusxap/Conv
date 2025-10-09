#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <random>    // Для случайных чисел
#include <algorithm> // Для std::generate
#include <cmath>     // Для std::abs

// Скалярная версия (a*b + c)
void scalar_fma(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c)
{
    for (size_t i = 0; i < a.size(); ++i)
    {
        c[i] = a[i] * b[i] + c[i];
    }
}

// // Векторная версия (AVX + FMA)
// void avx_fma(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c)
// {
//     size_t i = 0;
//     for (; i + 7 < a.size(); i += 8)
//     {
//         __m256 a_vec = _mm256_loadu_ps(&a[i]);
//         __m256 b_vec = _mm256_loadu_ps(&b[i]);
//         __m256 c_vec = _mm256_loadu_ps(&c[i]);

//         // Вычисляем (a * b) + c за одну инструкцию
//         c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

//         _mm256_storeu_ps(&c[i], c_vec);
//     }

//     // Остаток
//     for (; i < a.size(); ++i)
//     {
//         c[i] = a[i] * b[i] + c[i];
//     }
// }

int main()
{
    const size_t size = 100000000;
    const int warmup_runs = 3;
    const int bench_runs = 10;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> c_initial(size);

    // --- Инициализация случайными значениями ---
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    std::cout << "Generating random data..." << std::endl;
    std::generate(a.begin(), a.end(), [&]()
                  { return dis(gen); });
    std::generate(b.begin(), b.end(), [&]()
                  { return dis(gen); });
    std::generate(c_initial.begin(), c_initial.end(), [&]()
                  { return dis(gen); });
    std::cout << "Data generated." << std::endl;

    std::vector<float> result_scalar(size);
    std::vector<float> result_avx(size);

    // --- Тест скалярной версии ---
    std::cout << "\n--- Benchmarking Scalar FMA ---" << std::endl;
    // Прогрев
    for (int i = 0; i < warmup_runs; ++i)
    {
        result_scalar = c_initial;
        scalar_fma(a, b, result_scalar);
    }
    // Измерение
    double scalar_total_time = 0.0;
    for (int i = 0; i < bench_runs; ++i)
    {
        result_scalar = c_initial;
        auto start = std::chrono::high_resolution_clock::now();
        scalar_fma(a, b, result_scalar);
        auto end = std::chrono::high_resolution_clock::now();
        scalar_total_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double scalar_avg_time = scalar_total_time / bench_runs;
    std::cout << "Average Scalar FMA time: " << scalar_avg_time << " ms" << std::endl;

    // // --- Тест AVX версии ---
    // std::cout << "\n--- Benchmarking AVX FMA ---" << std::endl;
    // // Прогрев
    // for (int i = 0; i < warmup_runs; ++i)
    // {
    //     result_avx = c_initial;
    //     avx_fma(a, b, result_avx);
    // }
    // // Измерение
    // double avx_total_time = 0.0;
    // for (int i = 0; i < bench_runs; ++i)
    // {
    //     result_avx = c_initial;
    //     auto start = std::chrono::high_resolution_clock::now();
    //     avx_fma(a, b, result_avx);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     avx_total_time += std::chrono::duration<double, std::milli>(end - start).count();
    // }
    // double avx_avg_time = avx_total_time / bench_runs;
    // std::cout << "Average AVX FMA time:    " << avx_avg_time << " ms" << std::endl;

    // // --- Результаты ---
    // std::cout << "\nSpeedup: " << scalar_avg_time / avx_avg_time << "x" << std::endl;

    return 0;
}