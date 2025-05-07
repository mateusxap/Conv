#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>

// Структура для хранения тензора
struct Tensor4D {
    std::vector<float> data;
    size_t B_dim, C_dim, H_dim, W_dim;

    Tensor4D(size_t b = 0, size_t c = 0, size_t h = 0, size_t w = 0)
        : B_dim(b), C_dim(c), H_dim(h), W_dim(w), data(b* c* h* w, 0.0f) {}

    inline float& operator()(size_t n, size_t c, size_t h, size_t w) {
        return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
    }
    inline const float& operator()(size_t n, size_t c, size_t h, size_t w) const {
        return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
    }
    void setRandom(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
        for (float& val : data) { val = dist(rng); }
    }
    size_t size() const { return data.size(); }
    void fill(float val) { std::fill(data.begin(), data.end(), val); }
};

// Вспомогательные функции
// Паддинг остается как отдельная утилита
Tensor4D pad_same_3x3(const Tensor4D& input) {
    Tensor4D padded_output(input.B_dim, input.C_dim, input.H_dim + 2, input.W_dim + 2);
    for (size_t n = 0; n < input.B_dim; ++n) {
        for (size_t c = 0; c < input.C_dim; ++c) {
            for (size_t h = 0; h < input.H_dim; ++h) {
                for (size_t w = 0; w < input.W_dim; ++w) {
                    padded_output(n, c, h + 1, w + 1) = input(n, c, h, w);
                }
            }
        }
    }
    return padded_output;
}

Tensor4D concatenateChannels(const Tensor4D& t1, const Tensor4D& t2) {
    assert(t1.B_dim == t2.B_dim && t1.H_dim == t2.H_dim && t1.W_dim == t2.W_dim);
    size_t N = t1.B_dim, C1 = t1.C_dim, C2 = t2.C_dim, H = t1.H_dim, W = t1.W_dim;
    size_t C_total = C1 + C2;
    Tensor4D result(N, C_total, H, W);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C1; ++c) result(n, c, h, w) = t1(n, c, h, w);
                for (size_t c = 0; c < C2; ++c) result(n, C1 + c, h, w) = t2(n, c, h, w);
            }
        }
    }
    return result;
}

// Реализация сверток

// Ожидает на вход тензор с уже примененным паддингом, если он нужен.
Tensor4D convolve_basic(const Tensor4D& input_maybe_padded, const Tensor4D& kernel) {
    const size_t N = input_maybe_padded.B_dim;
    const size_t C_in = input_maybe_padded.C_dim;
    const size_t H_in = input_maybe_padded.H_dim; // Высота входа (может быть с паддингом)
    const size_t W_in = input_maybe_padded.W_dim; // Ширина входа (может быть с паддингом)
    const size_t C_out = kernel.C_dim;
    const size_t KH = kernel.H_dim;
    const size_t KW = kernel.W_dim;

    // Рассчитываем выходные размеры (стандартная формула, stride=1)
    const size_t H_out = H_in - KH + 1;
    const size_t W_out = W_in - KW + 1;

    Tensor4D output(N, C_out, H_out, W_out); // Создаем выходной тензор

    for (size_t n = 0; n < N; ++n) {                 // Batch
        for (size_t c_out = 0; c_out < C_out; ++c_out) { // Output Channel
            for (size_t h_out = 0; h_out < H_out; ++h_out) { // Output Height
                for (size_t w_out = 0; w_out < W_out; ++w_out) { // Output Width
                    float accumulator = 0.0f;
                    for (size_t c_in = 0; c_in < C_in; ++c_in) { // Input Channel
                        for (size_t kh = 0; kh < KH; ++kh) {     // Kernel Height
                            for (size_t kw = 0; kw < KW; ++kw) { // Kernel Width
                                size_t kernel_idx = c_out * (C_in * KH * KW) + c_in * (KH * KW) + kh * KW + kw;
                                // Доступ к входному тензору (который может быть паддированным)
                                accumulator += input_maybe_padded(n, c_in, h_out + kh, w_out + kw) * kernel.data[kernel_idx];
                            }
                        }
                    }
                    output(n, c_out, h_out, w_out) = accumulator;
                }
            }
        }
    }
    return output;
}


// Метод C: Объединенная свертка с 'if' внутри цикла
Tensor4D convolve_fused_1x1_3x3_with_if(const Tensor4D& input,
    const Tensor4D& kernel_1x1,
    const Tensor4D& kernel_3x3)
{
    const size_t N = input.B_dim;
    const size_t C_in = input.C_dim;
    const size_t H_in = input.H_dim; // Оригинальная высота
    const size_t W_in = input.W_dim; // Оригинальная ширина
    const size_t C_out_1x1 = kernel_1x1.C_dim;
    const size_t C_out_3x3 = kernel_3x3.C_dim;
    const size_t C_out_total = C_out_1x1 + C_out_3x3;

    // Паддинг выполняется внутри этой функции, т.к. она "fused"
    Tensor4D padded_input = pad_same_3x3(input);
    Tensor4D output(N, C_out_total, H_in, W_in); // Выходной размер как у оригинала

    for (size_t n = 0; n < N; ++n) {
        for (size_t h_out = 0; h_out < H_in; ++h_out) { // Цикл по оригинальным размерам
            for (size_t w_out = 0; w_out < W_in; ++w_out) {
                for (size_t c_in = 0; c_in < C_in; ++c_in) {
                    for (size_t kh = 0; kh < 3; ++kh) {
                        for (size_t kw = 0; kw < 3; ++kw) {
                            float input_val = padded_input(n, c_in, h_out + kh, w_out + kw);

                            // 3x3 part
                            size_t kernel_3x3_base_idx = c_in * 9 + kh * 3 + kw;
                            for (size_t c_out3 = 0; c_out3 < C_out_3x3; ++c_out3) {
                                size_t kernel_3x3_full_idx = c_out3 * (C_in * 9) + kernel_3x3_base_idx;
                                output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val * kernel_3x3.data[kernel_3x3_full_idx];
                            }

                            // 1x1 part (conditional)
                            if (kh == 1 && kw == 1) {
                                size_t kernel_1x1_base_idx = c_in;
                                for (size_t c_out1 = 0; c_out1 < C_out_1x1; ++c_out1) {
                                    size_t kernel_1x1_full_idx = c_out1 * C_in + kernel_1x1_base_idx;
                                    output(n, c_out1, h_out, w_out) += input_val * kernel_1x1.data[kernel_1x1_full_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}


// Метод D: Объединенная свертка без if
Tensor4D convolve_fused_1x1_3x3_no_if(const Tensor4D& input,
    const Tensor4D& kernel_1x1,
    const Tensor4D& kernel_3x3)
{
    const size_t N = input.B_dim;
    const size_t C_in = input.C_dim;
    const size_t H_in = input.H_dim;
    const size_t W_in = input.W_dim;
    const size_t C_out_1x1 = kernel_1x1.C_dim;
    const size_t C_out_3x3 = kernel_3x3.C_dim;
    const size_t C_out_total = C_out_1x1 + C_out_3x3;

    Tensor4D padded_input = pad_same_3x3(input); // Паддинг 
    Tensor4D output(N, C_out_total, H_in, W_in); // Выходной размер как у оригинала

    for (size_t n = 0; n < N; ++n) {
        for (size_t h_out = 0; h_out < H_in; ++h_out) { // Цикл по оригинальным размерам
            for (size_t w_out = 0; w_out < W_in; ++w_out) {
                for (size_t c_in = 0; c_in < C_in; ++c_in) {

                    // 3x3 part
                    for (size_t kh = 0; kh < 3; ++kh) {
                        for (size_t kw = 0; kw < 3; ++kw) {
                            float input_val_3x3 = padded_input(n, c_in, h_out + kh, w_out + kw);
                            size_t kernel_3x3_base_idx = c_in * 9 + kh * 3 + kw;
                            for (size_t c_out3 = 0; c_out3 < C_out_3x3; ++c_out3) {
                                size_t kernel_3x3_full_idx = c_out3 * (C_in * 9) + kernel_3x3_base_idx;
                                output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val_3x3 * kernel_3x3.data[kernel_3x3_full_idx];
                            }
                        }
                    }

                    // 1x1 part
                    float input_val_center = padded_input(n, c_in, h_out + 1, w_out + 1);
                    size_t kernel_1x1_base_idx = c_in;
                    for (size_t c_out1 = 0; c_out1 < C_out_1x1; ++c_out1) {
                        size_t kernel_1x1_full_idx = c_out1 * C_in + kernel_1x1_base_idx;
                        output(n, c_out1, h_out, w_out) += input_val_center * kernel_1x1.data[kernel_1x1_full_idx];
                    }

                }
            }
        }
    }
    return output;
}

// Функция для проверки близости тензоров
double check_difference(const Tensor4D& t1, const Tensor4D& t2) {
    if (t1.size() != t2.size() || t1.size() == 0) { return -1.0; }
    double diff_sum = 0.0;
    for (size_t i = 0; i < t1.data.size(); ++i) {
        diff_sum += std::fabs(t1.data[i] - t2.data[i]);
    }
    return diff_sum / t1.size();
}


int main() {
    // Параметры
    const size_t N_BATCH = 4;
    const size_t H_DIM = 28;
    const size_t W_DIM = 28;
    const size_t C_IN_DIM = 64;
    const size_t C_OUT_1x1_DIM = 32;
    const size_t C_OUT_3x3_DIM = 48;
    const size_t C_OUT_TOTAL_DIM = C_OUT_1x1_DIM + C_OUT_3x3_DIM;
    const int N_ITERATIONS = 100;
    const int WARMUP_ITERATIONS = 5;

    // Инициализация данных
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Tensor4D input(N_BATCH, C_IN_DIM, H_DIM, W_DIM);
    input.setRandom(rng, dist);

    // Ядра
    Tensor4D kernel_1x1(1, C_OUT_1x1_DIM, 1, 1);
    kernel_1x1.data.resize(C_OUT_1x1_DIM * C_IN_DIM);
    for (float& v : kernel_1x1.data) v = dist(rng);

    Tensor4D kernel_3x3(1, C_OUT_3x3_DIM, 3, 3);
    kernel_3x3.data.resize(C_OUT_3x3_DIM * C_IN_DIM * 9);
    for (float& v : kernel_3x3.data) v = dist(rng);

    // Ядро для метода B
    Tensor4D kernel_combined_zeros(1, C_OUT_TOTAL_DIM, 3, 3);
    kernel_combined_zeros.data.resize(C_OUT_TOTAL_DIM * C_IN_DIM * 9);
    for (size_t c_out = 0; c_out < C_OUT_1x1_DIM; ++c_out) {
        for (size_t c_in = 0; c_in < C_IN_DIM; ++c_in) {
            kernel_combined_zeros.data[c_out * (C_IN_DIM * 9) + c_in * 9 + 4] = kernel_1x1.data[c_out * C_IN_DIM + c_in];
        }
    }
    for (size_t c_out = 0; c_out < C_OUT_3x3_DIM; ++c_out) {
        for (size_t c_in = 0; c_in < C_IN_DIM; ++c_in) {
            for (size_t k = 0; k < 9; ++k) {
                size_t combined_idx = (C_OUT_1x1_DIM + c_out) * (C_IN_DIM * 9) + c_in * 9 + k;
                size_t kernel_3x3_idx = c_out * (C_IN_DIM * 9) + c_in * 9 + k;
                kernel_combined_zeros.data[combined_idx] = kernel_3x3.data[kernel_3x3_idx];
            }
        }
    }

    // Переменные для хранения результатов и времени
    Tensor4D output_A, output_B, output_C, output_D;
    double total_duration_A = 0, total_duration_B = 0, total_duration_C = 0, total_duration_D = 0;

    // Прогрев
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        // Метод A
        Tensor4D out1 = convolve_basic(input, kernel_1x1); // 1x1 на оригинальном входе
        Tensor4D input_padded_A = pad_same_3x3(input);     // Паддинг для 3x3
        Tensor4D out2 = convolve_basic(input_padded_A, kernel_3x3); // 3x3 на паддированном
        output_A = concatenateChannels(out1, out2);

        // Метод B
        Tensor4D input_padded_B = pad_same_3x3(input); // Паддинг для 3x3
        output_B = convolve_basic(input_padded_B, kernel_combined_zeros); // 3x3 на паддированном

        //Методы C и D (паддинг внутри)
        output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
        output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
    }

    // Замеры времени
    std::cout << "Starting benchmarks (" << N_ITERATIONS << " iterations)..." << std::endl;
    std::string method_A_name = "Method A (Separate basic_conv + Concat)";
    std::string method_B_name = "Method B (Combined basic_conv Zeros) ";
    std::string method_C_name = "Method C (Fused 1x1 & 3x3, with If)  ";
    std::string method_D_name = "Method D (Fused 1x1 & 3x3, no If)    ";

    // Метод A
    for (int i = 0; i < N_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor4D out1 = convolve_basic(input, kernel_1x1);
        Tensor4D input_padded_A = pad_same_3x3(input); // Паддинг перед 3x3
        Tensor4D out2 = convolve_basic(input_padded_A, kernel_3x3);
        auto end = std::chrono::high_resolution_clock::now();
        output_A = concatenateChannels(out1, out2);
        total_duration_A += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_duration_A = total_duration_A / N_ITERATIONS;
    std::cout << method_A_name << " Average Time: " << avg_duration_A << " ms" << std::endl;

    // Метод B
    for (int i = 0; i < N_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor4D input_padded_B = pad_same_3x3(input); // Паддинг перед 3x3
        output_B = convolve_basic(input_padded_B, kernel_combined_zeros);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_B += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_duration_B = total_duration_B / N_ITERATIONS;
    std::cout << method_B_name << " Average Time: " << avg_duration_B << " ms" << std::endl;

    // Метод C (Fused with If)
    for (int i = 0; i < N_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_C += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_duration_C = total_duration_C / N_ITERATIONS;
    std::cout << method_C_name << " Average Time: " << avg_duration_C << " ms" << std::endl;

    // Метод D (Fused no If)
    for (int i = 0; i < N_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_D += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_duration_D = total_duration_D / N_ITERATIONS;
    std::cout << method_D_name << " Average Time: " << avg_duration_D << " ms" << std::endl;


    // Проверка на близость результатов
    std::cout << "\nChecking differences..." << std::endl;
    // Пересчитываем выход A один раз для финальной проверки
    Tensor4D out1_final = convolve_basic(input, kernel_1x1);
    Tensor4D input_padded_final = pad_same_3x3(input);
    Tensor4D out2_final = convolve_basic(input_padded_final, kernel_3x3);
    output_A = concatenateChannels(out1_final, out2_final);
    // Пересчитываем B, C, D для чистоты сравнения
    output_B = convolve_basic(input_padded_final, kernel_combined_zeros);
    output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
    output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);

    double diff_AB = check_difference(output_A, output_B);
    double diff_AC = check_difference(output_A, output_C);
    double diff_AD = check_difference(output_A, output_D);
    std::cout << "Average absolute difference (A vs B): " << diff_AB << std::endl;
    std::cout << "Average absolute difference (A vs C): " << diff_AC << std::endl;
    std::cout << "Average absolute difference (A vs D): " << diff_AD << std::endl;

    return 0;
}