#include "Conv.hpp"

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

int benchmark_NCHWc_conv(size_t N_BATCH, size_t H_DIM, size_t W_DIM, size_t C_IN_DIM, size_t C_OUT_DIM, size_t KH, size_t KW, int N_ITERATIONS = 10, int WARMUP_ITERATIONS = 5)
{
	// init
	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> input_NCHWc(N_BATCH * C_IN_DIM * H_DIM * W_DIM);
	for (float &val : input_NCHWc)
	{
		val = dist(rng);
	}
	std::vector<float> kernel_OIHWio(C_OUT_DIM * C_IN_DIM * KH * KW);
	for (float &val : kernel_OIHWio)
	{
		val = dist(rng);
	}

	float total_duration = 0;
	// Прогрев
	std::cout << "Warming up..." << std::endl;
	for (int i = 0; i < WARMUP_ITERATIONS; ++i)
	{
		auto output = conv_optimized(input_NCHWc, kernel_OIHWio, N_BATCH, C_IN_DIM, H_DIM, W_DIM, C_OUT_DIM, KH, KW);
	}

	// Замеры времени
	std::cout << "Starting benchmarks (" << N_ITERATIONS << " iterations)..." << std::endl;
	for (int i = 0; i < N_ITERATIONS; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();
		auto output = conv_optimized(input_NCHWc, kernel_OIHWio, N_BATCH, C_IN_DIM, H_DIM, W_DIM, C_OUT_DIM, KH, KW);
		auto end = std::chrono::high_resolution_clock::now();
		total_duration += std::chrono::duration<double, std::milli>(end - start).count();
	}
	auto avg_duration = total_duration / N_ITERATIONS;
	std::cout << "Optimized NCHWc Conv Average Time: " << avg_duration << " ms" << std::endl;

	return 0;
}

int main()
{
	// Параметры
	const size_t N_BATCH = 1;
	const size_t C_IN_DIM = 1024;
	const size_t H_DIM = 28;
	const size_t W_DIM = 28;
	const size_t C_OUT_1x1_DIM = 64;
	const size_t C_OUT_3x3_DIM = 48;
	const int N_ITERATIONS = 50;
	const int WARMUP_ITERATIONS = 10;

	// benchmark_NCHW_convs(N_BATCH, H_DIM, W_DIM, C_IN_DIM, C_OUT_1x1_DIM, C_OUT_3x3_DIM, N_ITERATIONS, WARMUP_ITERATIONS);
	// benchmark_NHWC_convs(N_BATCH, H_DIM, W_DIM, C_IN_DIM, C_OUT_1x1_DIM, C_OUT_3x3_DIM, N_ITERATIONS, WARMUP_ITERATIONS);
	benchmark_NCHWc_conv(N_BATCH, H_DIM, W_DIM, C_IN_DIM, C_OUT_3x3_DIM, 3, 3, N_ITERATIONS, WARMUP_ITERATIONS);
	return 0;
}