#include "Conv.hpp"
#include <iomanip>
#include "omp.h"
#include <algorithm>
#include <vector>

// int benchmark_NCHW_convs(size_t N_BATCH, size_t H_DIM, size_t W_DIM, size_t C_IN_DIM, size_t C_OUT_1x1_DIM, size_t C_OUT_3x3_DIM, int N_ITERATIONS = 10, int WARMUP_ITERATIONS = 5)
// {
// 	// init
// 	std::mt19937 rng(1234);
// 	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
// 	const size_t C_OUT_TOTAL_DIM = C_OUT_1x1_DIM + C_OUT_3x3_DIM;

// 	Tensor4D_NCHW input(N_BATCH, C_IN_DIM, H_DIM, W_DIM);
// 	input.setRandom(rng, dist);

// 	// Kernels
// 	Tensor4D_NCHW kernel_1x1(C_OUT_1x1_DIM, C_IN_DIM, 1, 1);
// 	kernel_1x1.setRandom(rng, dist);

// 	Tensor4D_NCHW kernel_3x3(C_OUT_3x3_DIM, C_IN_DIM, 3, 3);
// 	kernel_3x3.setRandom(rng, dist);

// 	// Big kernel for B
// 	Tensor4D_NCHW kernel_combined_zeros(C_OUT_TOTAL_DIM, C_IN_DIM, 3, 3);
// 	for (size_t c_out = 0; c_out < C_OUT_1x1_DIM; ++c_out)
// 	{
// 		for (size_t c_in = 0; c_in < C_IN_DIM; ++c_in)
// 		{
// 			// Cental el 3x3: (1,1)
// 			kernel_combined_zeros(c_out, c_in, 1, 1) = kernel_1x1(c_out, c_in, 0, 0);
// 		}
// 	}
// 	for (size_t c_out = 0; c_out < C_OUT_3x3_DIM; ++c_out)
// 	{
// 		for (size_t c_in = 0; c_in < C_IN_DIM; ++c_in)
// 		{
// 			for (size_t kh = 0; kh < 3; ++kh)
// 			{
// 				for (size_t kw = 0; kw < 3; ++kw)
// 				{
// 					// Сдвиг по выходным каналам для 3x3
// 					kernel_combined_zeros(C_OUT_1x1_DIM + c_out, c_in, kh, kw) =
// 						kernel_3x3(c_out, c_in, kh, kw);
// 				}
// 			}
// 		}
// 	}

// 	// Переменные для хранения результатов и времени
// 	Tensor4D_NCHW output_A, output_B, output_C, output_D;
// 	double total_duration_A = 0, total_duration_B = 0, total_duration_C = 0, total_duration_D = 0;

// 	// Прогрев
// 	std::cout << "Warming up..." << std::endl;
// 	for (int i = 0; i < WARMUP_ITERATIONS; ++i)
// 	{
// 		// Метод A
// 		Tensor4D_NCHW out1 = convolve_basic(input, kernel_1x1);			 // 1x1 на оригинальном входе
// 		Tensor4D_NCHW input_padded_A = input.pad_same_3x3();			 // Паддинг для 3x3
// 		Tensor4D_NCHW out2 = convolve_basic(input_padded_A, kernel_3x3); // 3x3 на паддированном
// 		output_A = out1.concatenateChannels(out2);

// 		// Метод B
// 		Tensor4D_NCHW input_padded_B = input.pad_same_3x3();			  // Паддинг для 3x3
// 		output_B = convolve_basic(input_padded_B, kernel_combined_zeros); // 3x3 на паддированном

// 		// Методы C и D (паддинг внутри)
// 		output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
// 		output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
// 	}

// 	// Замеры времени
// 	std::cout << "Starting benchmarks (" << N_ITERATIONS << " iterations)..." << std::endl;
// 	std::string method_A_name = "Method A (Separate basic_conv + Concat)";
// 	std::string method_B_name = "Method B (Combined basic_conv Zeros) ";
// 	std::string method_C_name = "Method C (Fused 1x1 & 3x3, with If)  ";
// 	std::string method_D_name = "Method D (Fused 1x1 & 3x3, no If)    ";

// 	// Метод A
// 	for (int i = 0; i < N_ITERATIONS; ++i)
// 	{
// 		auto start = std::chrono::high_resolution_clock::now();
// 		Tensor4D_NCHW out1 = convolve_basic(input, kernel_1x1);
// 		Tensor4D_NCHW input_padded_A = input.pad_same_3x3(); // Паддинг перед 3x3
// 		Tensor4D_NCHW out2 = convolve_basic(input_padded_A, kernel_3x3);
// 		auto end = std::chrono::high_resolution_clock::now();
// 		output_A = out1.concatenateChannels(out2);
// 		total_duration_A += std::chrono::duration<double, std::milli>(end - start).count();
// 	}
// 	double avg_duration_A = total_duration_A / N_ITERATIONS;
// 	std::cout << method_A_name << " Average Time: " << avg_duration_A << " ms" << std::endl;

// 	// Метод B
// 	for (int i = 0; i < N_ITERATIONS; ++i)
// 	{
// 		auto start = std::chrono::high_resolution_clock::now();
// 		Tensor4D_NCHW input_padded_B = input.pad_same_3x3(); // Паддинг перед 3x3
// 		output_B = convolve_basic(input_padded_B, kernel_combined_zeros);
// 		auto end = std::chrono::high_resolution_clock::now();
// 		total_duration_B += std::chrono::duration<double, std::milli>(end - start).count();
// 	}
// 	double avg_duration_B = total_duration_B / N_ITERATIONS;
// 	std::cout << method_B_name << " Average Time: " << avg_duration_B << " ms" << std::endl;

// 	// Метод C (Fused with If)
// 	for (int i = 0; i < N_ITERATIONS; ++i)
// 	{
// 		auto start = std::chrono::high_resolution_clock::now();
// 		output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
// 		auto end = std::chrono::high_resolution_clock::now();
// 		total_duration_C += std::chrono::duration<double, std::milli>(end - start).count();
// 	}
// 	double avg_duration_C = total_duration_C / N_ITERATIONS;
// 	std::cout << method_C_name << " Average Time: " << avg_duration_C << " ms" << std::endl;

// 	// Метод D (Fused no If)
// 	for (int i = 0; i < N_ITERATIONS; ++i)
// 	{
// 		auto start = std::chrono::high_resolution_clock::now();
// 		output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
// 		auto end = std::chrono::high_resolution_clock::now();
// 		total_duration_D += std::chrono::duration<double, std::milli>(end - start).count();
// 	}
// 	double avg_duration_D = total_duration_D / N_ITERATIONS;
// 	std::cout << method_D_name << " Average Time: " << avg_duration_D << " ms" << std::endl;

// 	// Проверка на близость результатов
// 	std::cout << "\nChecking differences..." << std::endl;
// 	// Пересчитываем выход A один раз для финальной проверки
// 	Tensor4D_NCHW out1_final = convolve_basic(input, kernel_1x1);
// 	Tensor4D_NCHW input_padded_final = input.pad_same_3x3();
// 	Tensor4D_NCHW out2_final = convolve_basic(input_padded_final, kernel_3x3);
// 	output_A = out1_final.concatenateChannels(out2_final);
// 	// Пересчитываем B, C, D для чистоты сравнения
// 	output_B = convolve_basic(input_padded_final, kernel_combined_zeros);
// 	output_C = convolve_fused_1x1_3x3_with_if(input, kernel_1x1, kernel_3x3);
// 	output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);

// 	double diff_AB = output_A.check_difference(output_B);
// 	double diff_AC = output_A.check_difference(output_C);
// 	double diff_AD = output_A.check_difference(output_D);
// 	std::cout << "Average absolute difference (A vs B): " << diff_AB << std::endl;
// 	std::cout << "Average absolute difference (A vs C): " << diff_AC << std::endl;
// 	std::cout << "Average absolute difference (A vs D): " << diff_AD << std::endl;

// 	return 0;
// }

// int benchmark_NHWC_convs(size_t N_BATCH, size_t H_DIM, size_t W_DIM, size_t C_IN_DIM, size_t C_OUT_1x1_DIM, size_t C_OUT_3x3_DIM, int N_ITERATIONS = 10, int WARMUP_ITERATIONS = 5)
// {
// 	// init
// 	std::mt19937 rng(1234);
// 	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

// 	Tensor4D_NHWC input(N_BATCH, H_DIM, W_DIM, C_IN_DIM);
// 	input.setRandom(rng, dist);

// 	// Kernels
// 	Tensor4D_HWIO kernel_1x1(1, 1, C_OUT_1x1_DIM, C_IN_DIM);
// 	kernel_1x1.setRandom(rng, dist);

// 	Tensor4D_HWIO kernel_3x3(3, 3, C_OUT_3x3_DIM, C_IN_DIM);
// 	kernel_3x3.setRandom(rng, dist);

// 	// Переменные для хранения результатов и времени
// 	Tensor4D_NHWC output_A, output_D;
// 	double total_duration_A = 0, total_duration_D = 0;

// 	// Прогрев
// 	std::cout << "Warming up..." << std::endl;
// 	for (int i = 0; i < WARMUP_ITERATIONS; ++i)
// 	{
// 		Tensor4D_NHWC out1 = convolve_basic(input, kernel_1x1);			 // 1x1 на оригинальном входе
// 		Tensor4D_NHWC input_padded_A = input.pad_same_3x3();			 // Паддинг для 3x3
// 		Tensor4D_NHWC out2 = convolve_basic(input_padded_A, kernel_3x3); // 3x3 на паддированном
// 		output_A = out1.concatenateChannels(out2);
// 		// output_D = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
// 	}

// 	// Замеры времени
// 	std::cout << "Starting benchmarks (" << N_ITERATIONS << " iterations)..." << std::endl;
// 	std::string method_A_name_NHWC = "Method A NHWC (Separate basic_conv + Concat)";

// 	// Метод A
// 	for (int i = 0; i < N_ITERATIONS; ++i)
// 	{
// 		auto start = std::chrono::high_resolution_clock::now();
// 		Tensor4D_NHWC out1 = convolve_basic(input, kernel_1x1);			 // 1x1 на оригинальном входе
// 		Tensor4D_NHWC input_padded_A = input.pad_same_3x3();			 // Паддинг для 3x3
// 		Tensor4D_NHWC out2 = convolve_basic(input_padded_A, kernel_3x3); // 3x3 на паддированном
// 		auto end = std::chrono::high_resolution_clock::now();
// 		output_A = out1.concatenateChannels(out2);
// 		total_duration_A += std::chrono::duration<double, std::milli>(end - start).count();
// 	}
// 	auto avg_duration_A = total_duration_A / N_ITERATIONS;
// 	std::cout << method_A_name_NHWC << " Average Time: " << avg_duration_A << " ms" << std::endl;

// 	// // Метод D (Fused no If)
// 	// for (int i = 0; i < N_ITERATIONS; ++i)
// 	// {
// 	// 	auto start = std::chrono::high_resolution_clock::now();
// 	// 	output_D_NHWC = convolve_fused_1x1_3x3_no_if(input, kernel_1x1, kernel_3x3);
// 	// 	auto end = std::chrono::high_resolution_clock::now();
// 	// 	total_duration_D += std::chrono::duration<double, std::milli>(end - start).count();
// 	// }
// 	// auto avg_duration_D = total_duration_D_NHWC / N_ITERATIONS;
// 	// std::cout << method_D_name << " Average Time: " << avg_duration_D << " ms" << std::endl;

// 	// diff_AD = check_difference(output_A_NHWC, output_D_NHWC);
// 	// std::cout << "Average absolute difference (A vs D): " << diff_AD << std::endl;
// 	return 0;
// }

int benchmark_NCHWc_conv()
{
	// init
	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> output(N * H_out * W_out * C_out);
	std::vector<float> input_NCHWc(N * C_in * H_in * W_in);
	for (float &val : input_NCHWc)
	{
		val = dist(rng);
	}
	std::vector<float> kernel_OIHWio(C_out * C_in * KH * KW);
	for (float &val : kernel_OIHWio)
	{
		val = dist(rng);
	}

	std::vector<double> durations;
	durations.reserve(N_ITERATIONS);
	// Прогрев
	std::cout << "Warming up..." << std::endl;
	for (int i = 0; i < WARMUP_ITERATIONS; ++i)
	{
		conv_optimized_w_c(input_NCHWc, kernel_OIHWio, output, C_out);
	}

	// Замеры времени
	std::cout << "Starting benchmarks (" << N_ITERATIONS << " iterations)..." << std::endl;
	for (int i = 0; i < N_ITERATIONS; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();
		conv_optimized_w_c(input_NCHWc, kernel_OIHWio, output, C_out);
		auto end = std::chrono::high_resolution_clock::now();
		durations.push_back(std::chrono::duration<double, std::milli>(end - start).count());
	}

	std::sort(durations.begin(), durations.end());
	double median_duration;
	if (N_ITERATIONS % 2 == 0)
	{
		median_duration = (durations[N_ITERATIONS / 2 - 1] + durations[N_ITERATIONS / 2]) / 2.0;
	}
	else
	{
		median_duration = durations[N_ITERATIONS / 2];
	}
	std::cout << "Optimized NCHWc Conv Median Time: " << median_duration << " ms" << std::endl;

	return 0;
}

int benchmark_NCHWc_convs_googlenet()
{
	// init
	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// Input
	std::vector<float> input_NCHWc(N * C_in * H_in * W_in);
	for (float &val : input_NCHWc)
	{
		val = dist(rng);
	}

	// Sizes
	size_t C_out_1 = 384;
	size_t C_out_2 = 192;
	size_t C_out_3 = 48;
	size_t C_out_combined = C_out_1 + C_out_2 + C_out_3; // 624

	// Kernels
	std::vector<float> kernel1(C_out_1 * C_in * KH * KW);
	for (float &val : kernel1)
		val = dist(rng);

	std::vector<float> kernel2(C_out_2 * C_in * KH * KW);
	for (float &val : kernel2)
		val = dist(rng);

	std::vector<float> kernel3(C_out_3 * C_in * KH * KW);
	for (float &val : kernel3)
		val = dist(rng);

	std::vector<float> kernel_combined(C_out_combined * C_in * KH * KW);
	for (float &val : kernel_combined)
		val = dist(rng);

	// Outputs
	std::vector<float> output_combined(N * C_out_combined / BLOCK_SIZE * H_out * W_out * BLOCK_SIZE, 0.0f);
	std::vector<float> output1(N * C_out_1 / BLOCK_SIZE * H_out * W_out * BLOCK_SIZE, 0.0f);
	std::vector<float> output2(N * C_out_2 / BLOCK_SIZE * H_out * W_out * BLOCK_SIZE, 0.0f);
	std::vector<float> output3(N * C_out_3 / BLOCK_SIZE * H_out * W_out * BLOCK_SIZE, 0.0f);

	// -------------------------------------------------------
	// Correctness check: compare c_w and v3 against w_c
	// -------------------------------------------------------
	std::cout << "Running correctness checks..." << std::endl;

	// Reference: w_c
	std::vector<float> ref_combined(output_combined.size(), 0.0f);
	std::vector<float> ref1(output1.size(), 0.0f);
	std::vector<float> ref2(output2.size(), 0.0f);
	std::vector<float> ref3(output3.size(), 0.0f);
	conv_optimized_w_c(input_NCHWc, kernel_combined, ref_combined, C_out_combined);
	conv_optimized_w_c(input_NCHWc, kernel1, ref1, C_out_1);
	conv_optimized_w_c(input_NCHWc, kernel2, ref2, C_out_2);
	conv_optimized_w_c(input_NCHWc, kernel3, ref3, C_out_3);

	auto check_correctness = [](const std::string &name,
	                            const std::vector<float> &ref,
	                            const std::vector<float> &out) {
		const float tol = 1e-3f;
		for (size_t i = 0; i < ref.size(); ++i) {
			if (std::abs(ref[i] - out[i]) > tol) {
				std::cout << "[FAIL] " << name << " mismatch at index " << i
				          << ": ref=" << ref[i] << " got=" << out[i] << std::endl;
				return;
			}
		}
		std::cout << "[OK]   " << name << std::endl;
	};

	// Check c_w combined
	{
		std::vector<float> out(ref_combined.size(), 0.0f);
		conv_optimized_c_w(input_NCHWc, kernel_combined, out, C_out_combined);
		check_correctness("c_w  combined", ref_combined, out);
	}
	// Check c_w sequential
	{
		std::vector<float> o1(ref1.size(), 0.0f), o2(ref2.size(), 0.0f), o3(ref3.size(), 0.0f);
		conv_optimized_c_w(input_NCHWc, kernel1, o1, C_out_1);
		conv_optimized_c_w(input_NCHWc, kernel2, o2, C_out_2);
		conv_optimized_c_w(input_NCHWc, kernel3, o3, C_out_3);
		check_correctness("c_w  sequential kernel1", ref1, o1);
		check_correctness("c_w  sequential kernel2", ref2, o2);
		check_correctness("c_w  sequential kernel3", ref3, o3);
	}
	// Check v3 combined
	{
		std::vector<float> out(ref_combined.size(), 0.0f);
		conv_optimized_v3(input_NCHWc, kernel_combined, out, C_out_combined);
		check_correctness("v3   combined", ref_combined, out);
	}
	// Check v3 sequential
	{
		std::vector<float> o1(ref1.size(), 0.0f), o2(ref2.size(), 0.0f), o3(ref3.size(), 0.0f);
		conv_optimized_v3(input_NCHWc, kernel1, o1, C_out_1);
		conv_optimized_v3(input_NCHWc, kernel2, o2, C_out_2);
		conv_optimized_v3(input_NCHWc, kernel3, o3, C_out_3);
		check_correctness("v3   sequential kernel1", ref1, o1);
		check_correctness("v3   sequential kernel2", ref2, o2);
		check_correctness("v3   sequential kernel3", ref3, o3);
	}
	std::cout << std::endl;

	// -------------------------------------------------------
	// Benchmarking
	// -------------------------------------------------------
	auto benchmark = [&](const std::string &label, auto fn) -> double {
		std::vector<double> durations;
		durations.reserve(N_ITERATIONS);
		std::cout << "Warming up " << label << "..." << std::endl;
		for (int i = 0; i < WARMUP_ITERATIONS; ++i)
			fn();
		std::cout << "Benchmarking " << label << " (" << N_ITERATIONS << " iterations)..." << std::endl;
		for (int i = 0; i < N_ITERATIONS; ++i) {
			auto start = std::chrono::high_resolution_clock::now();
			fn();
			auto end = std::chrono::high_resolution_clock::now();
			durations.push_back(std::chrono::duration<double, std::milli>(end - start).count());
		}
		std::sort(durations.begin(), durations.end());
		if (durations.size() % 2 == 0)
			return (durations[durations.size() / 2 - 1] + durations[durations.size() / 2]) / 2.0;
		return durations[durations.size() / 2];
	};

	double med_wc_comb  = benchmark("w_c  combined",  [&]{ conv_optimized_w_c(input_NCHWc, kernel_combined, output_combined, C_out_combined); });
	double med_wc_seq   = benchmark("w_c  sequential", [&]{ conv_optimized_w_c(input_NCHWc, kernel1, output1, C_out_1); conv_optimized_w_c(input_NCHWc, kernel2, output2, C_out_2); conv_optimized_w_c(input_NCHWc, kernel3, output3, C_out_3); });

	double med_cw_comb  = benchmark("c_w  combined",  [&]{ conv_optimized_c_w(input_NCHWc, kernel_combined, output_combined, C_out_combined); });
	double med_cw_seq   = benchmark("c_w  sequential", [&]{ conv_optimized_c_w(input_NCHWc, kernel1, output1, C_out_1); conv_optimized_c_w(input_NCHWc, kernel2, output2, C_out_2); conv_optimized_c_w(input_NCHWc, kernel3, output3, C_out_3); });

	double med_v3_comb  = benchmark("v3   combined",  [&]{ conv_optimized_v3(input_NCHWc, kernel_combined, output_combined, C_out_combined); });
	double med_v3_seq   = benchmark("v3   sequential", [&]{ conv_optimized_v3(input_NCHWc, kernel1, output1, C_out_1); conv_optimized_v3(input_NCHWc, kernel2, output2, C_out_2); conv_optimized_v3(input_NCHWc, kernel3, output3, C_out_3); });

	// -------------------------------------------------------
	// Results
	// -------------------------------------------------------
	std::cout << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << std::left << std::setw(35) << "Variant"
	          << std::setw(16) << "Combined (ms)"
	          << std::setw(16) << "Sequential (ms)"
	          << "Seq/Comb" << std::endl;
	std::cout << std::string(75, '-') << std::endl;

	auto row = [](const std::string &name, double comb, double seq) {
		std::cout << std::left << std::setw(35) << name
		          << std::setw(16) << comb
		          << std::setw(16) << seq
		          << (seq / comb) << "x" << std::endl;
	};
	row("w_c  (baseline)",      med_wc_comb,  med_wc_seq);
	row("c_w  (loop swapped)",  med_cw_comb,  med_cw_seq);
	row("v3   (AVX2 + aligned)",med_v3_comb,  med_v3_seq);

	std::cout << std::string(75, '-') << std::endl;
	std::cout << std::endl;
	std::cout << "Speedup v3 vs w_c (combined):   " << med_wc_comb / med_v3_comb << "x" << std::endl;
	std::cout << "Speedup v3 vs w_c (sequential):  " << med_wc_seq  / med_v3_seq  << "x" << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	return 0;
}


void benchmark_NCHWc_sweep()
{
    const int SWEEP_ITERS   = 100;
    const int SWEEP_WARMUP  = 5;
    const int HW_vals[]   = {7, 14, 28, 56};
    const int Cin_vals[]  = {16, 32, 64, 128, 256};
    const int Cout_vals[] = {16, 32, 64, 128, 256};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Header
    const int W = 130;
    std::cout << std::endl;
    std::cout << std::string(W, '=') << std::endl;
    std::cout << "  Sweep benchmark: 1x1 conv, N=1, BLOCK_SIZE=8" << std::endl;
    std::cout << "  Combined = 1 call with full Cout;  Sequential = 2 calls with Cout/2 each" << std::endl;
    std::cout << "  Iterations=" << SWEEP_ITERS << "  Warmup=" << SWEEP_WARMUP << std::endl;
    std::cout << std::string(W, '=') << std::endl;
    std::cout << std::left
              << std::setw(5)  << "HW"
              << std::setw(5)  << "Cin"
              << std::setw(5)  << "Cout"
              << " | "
              << std::setw(10) << "w_c_c"
              << std::setw(10) << "w_c_s"
              << " | "
              << std::setw(10) << "c_w_c"
              << std::setw(10) << "c_w_s"
              << " | "
              << std::setw(10) << "v3_c"
              << std::setw(10) << "v3_s"
              << " | "
              << std::setw(8)  << "v3/w_c"
              << std::setw(8)  << "v3s/wcs"
              << std::setw(8)  << "s/c v3"
              << " | "
              << "chk" << std::endl;
    std::cout << std::string(W, '-') << std::endl;

    for (int hw : HW_vals) {
        for (int ci : Cin_vals) {
            for (int co : Cout_vals) {
                ConvParams p;
                p.N    = 1;
                p.C_in = ci;
                p.H_in = hw;
                p.W_in = hw;
                p.KH   = 1;
                p.KW   = 1;

                int h_o = p.H_out();
                int w_o = p.W_out();
                size_t in_sz  = (size_t)p.N * ci * hw * hw;
                size_t out_sz = (size_t)p.N * h_o * w_o * co;
                int co_half   = co / 2;
                size_t k_sz_full = (size_t)co * ci;
                size_t k_sz_half = (size_t)co_half * ci;
                size_t out_sz_half = (size_t)p.N * h_o * w_o * co_half;

                // Allocate
                std::vector<float> input(in_sz);
                for (float &v : input) v = dist(rng);

                std::vector<float> kernel_full(k_sz_full);
                for (float &v : kernel_full) v = dist(rng);
                std::vector<float> kernel_h1(k_sz_half);
                for (float &v : kernel_h1) v = dist(rng);
                std::vector<float> kernel_h2(k_sz_half);
                for (float &v : kernel_h2) v = dist(rng);

                // Output buffers
                std::vector<float> out_wc(out_sz, 0.0f);
                std::vector<float> out_cw(out_sz, 0.0f);
                std::vector<float> out_v3(out_sz, 0.0f);
                std::vector<float> out_h1(out_sz_half, 0.0f);
                std::vector<float> out_h2(out_sz_half, 0.0f);

                // Correctness check (combined)
                conv_param_w_c(p, input.data(), kernel_full.data(), out_wc.data(), co);
                conv_param_v3 (p, input.data(), kernel_full.data(), out_v3.data(), co);
                bool ok = true;
                for (size_t i = 0; i < out_sz; i++) {
                    if (std::abs(out_wc[i] - out_v3[i]) > 1e-3f) { ok = false; break; }
                }

                // Warmup
                for (int it = 0; it < SWEEP_WARMUP; it++) {
                    conv_param_w_c(p, input.data(), kernel_full.data(), out_wc.data(), co);
                    conv_param_c_w(p, input.data(), kernel_full.data(), out_cw.data(), co);
                    conv_param_v3 (p, input.data(), kernel_full.data(), out_v3.data(), co);
                    conv_param_w_c(p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_w_c(p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                    conv_param_c_w(p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_c_w(p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                    conv_param_v3 (p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_v3 (p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                }

                // Benchmark helper
                auto bench = [&](auto fn) -> double {
                    std::vector<double> durs;
                    durs.reserve(SWEEP_ITERS);
                    for (int it = 0; it < SWEEP_ITERS; it++) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        fn();
                        auto t1 = std::chrono::high_resolution_clock::now();
                        durs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    }
                    std::sort(durs.begin(), durs.end());
                    return durs[durs.size() / 2];
                };

                // Combined (1 call with full Cout)
                double wc_c = bench([&]{ conv_param_w_c(p, input.data(), kernel_full.data(), out_wc.data(), co); });
                double cw_c = bench([&]{ conv_param_c_w(p, input.data(), kernel_full.data(), out_cw.data(), co); });
                double v3_c = bench([&]{ conv_param_v3 (p, input.data(), kernel_full.data(), out_v3.data(), co); });

                // Sequential (2 calls with Cout/2)
                double wc_s = bench([&]{
                    conv_param_w_c(p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_w_c(p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                });
                double cw_s = bench([&]{
                    conv_param_c_w(p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_c_w(p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                });
                double v3_s = bench([&]{
                    conv_param_v3(p, input.data(), kernel_h1.data(), out_h1.data(), co_half);
                    conv_param_v3(p, input.data(), kernel_h2.data(), out_h2.data(), co_half);
                });

                std::cout << std::left << std::fixed << std::setprecision(4)
                          << std::setw(5)  << hw
                          << std::setw(5)  << ci
                          << std::setw(5)  << co
                          << " | "
                          << std::setw(10) << wc_c
                          << std::setw(10) << wc_s
                          << " | "
                          << std::setw(10) << cw_c
                          << std::setw(10) << cw_s
                          << " | "
                          << std::setw(10) << v3_c
                          << std::setw(10) << v3_s
                          << " | "
                          << std::setw(8)  << (wc_c / v3_c)
                          << std::setw(8)  << (wc_s / v3_s)
                          << std::setw(8)  << (v3_s / v3_c)
                          << " | "
                          << (ok ? "OK" : "FAIL") << std::endl;
            }
        }
    }
    std::cout << std::string(W, '=') << std::endl;
    std::cout << "Columns: _c = combined (1 call), _s = sequential (2 x Cout/2)" << std::endl;
    std::cout << "v3/w_c  = speedup of v3 combined over w_c combined" << std::endl;
    std::cout << "v3s/wcs = speedup of v3 sequential over w_c sequential" << std::endl;
    std::cout << "s/c v3  = ratio of v3 sequential to v3 combined (>1 means combined is faster)" << std::endl;
}

int main()
{
	// omp_set_num_threads(8);
	//   Параметры
	//  benchmark_NCHW_convs(N_BATCH, H_DIM, W_DIM, C_IN_DIM, C_OUT_1x1_DIM, C_OUT_3x3_DIM, N_ITERATIONS, WARMUP_ITERATIONS);
	//  benchmark_NHWC_convs(N_BATCH, H_DIM, W_DIM, C_IN_DIM, C_OUT_1x1_DIM, C_OUT_3x3_DIM, N_ITERATIONS, WARMUP_ITERATIONS);
	// benchmark_NCHWc_convs_googlenet();
	benchmark_NCHWc_sweep();
	return 0;
}