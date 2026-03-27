#include "Conv.hpp"
#include <immintrin.h> // для AVX/SSE

// ------ Tensor4D impl ------

// double Tensor4D::check_difference(const Tensor4D &t2)
// {
// 	if (this->size() != t2.size() || this->size() == 0)
// 	{
// 		return -1.0;
// 	}
// 	double diff_sum = 0.0;
// 	for (int i = 0; i < this->size(); ++i)
// 	{
// 		diff_sum += std::fabs(this->data[i] - t2.data[i]);
// 	}
// 	return diff_sum / this->size();
// }

// // ------ Tensor4D_NCHW impl ------

// inline float &Tensor4D_NCHW::operator()(int n, int c, int h, int w)
// {
// 	return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
// }

// inline const float &Tensor4D_NCHW::operator()(int n, int c, int h, int w) const
// {
// 	return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
// }

// Tensor4D_NCHW Tensor4D_NCHW::pad_same_3x3() const
// {
// 	Tensor4D_NCHW padded_output(this->B_dim, this->C_dim, this->H_dim + 2, this->W_dim + 2);
// 	for (int n = 0; n < this->B_dim; ++n)
// 	{
// 		for (int c = 0; c < this->C_dim; ++c)
// 		{
// 			for (int h = 0; h < this->H_dim; ++h)
// 			{
// 				for (int w = 0; w < this->W_dim; ++w)
// 				{
// 					padded_output(n, c, h + 1, w + 1) = (*this)(n, c, h, w);
// 				}
// 			}
// 		}
// 	}
// 	return padded_output;
// }

// Tensor4D_NCHW Tensor4D_NCHW::concatenateChannels(const Tensor4D &t2) const
// {
// 	assert(this->B_dim == t2.getB() && this->H_dim == t2.getH() && this->W_dim == t2.getW());
// 	int N = this->B_dim, C1 = this->C_dim, C2 = t2.getC(), H = this->H_dim, W = this->W_dim;
// 	int C_total = C1 + C2;
// 	Tensor4D_NCHW result(N, C_total, H, W);
// 	for (int n = 0; n < N; ++n)
// 	{
// 		for (int h = 0; h < H; ++h)
// 		{
// 			for (int w = 0; w < W; ++w)
// 			{
// 				for (int c = 0; c < C1; ++c)
// 					result(n, c, h, w) = (*this)(n, c, h, w);
// 				for (int c = 0; c < C2; ++c)
// 					result(n, C1 + c, h, w) = t2(n, c, h, w);
// 			}
// 		}
// 	}
// 	return result;
// }

// Tensor4D_NCHW convolve_basic(const Tensor4D_NCHW &input_maybe_padded, const Tensor4D_NCHW &kernel)
// {
// 	const int N = input_maybe_padded.getB();
// 	const int C_in = input_maybe_padded.getC();
// 	const int H_in = input_maybe_padded.getH();
// 	const int W_in = input_maybe_padded.getW();
// 	const int C_out = kernel.getB();
// 	const int KH = kernel.getH();
// 	const int KW = kernel.getW();
// 	const int H_out = H_in - KH + 1;
// 	const int W_out = W_in - KW + 1;

// 	Tensor4D_NCHW output(N, C_out, H_out, W_out);
// #pragma omp parallel for collapse(2)
// 	for (int n = 0; n < N; ++n)
// 	{ // Batch
// 		for (int c_out = 0; c_out < C_out; ++c_out)
// 		{ // Output Channel
// 			for (int h_out = 0; h_out < H_out; ++h_out)
// 			{ // Output Height
// 				for (int w_out = 0; w_out < W_out; ++w_out)
// 				{ // Output Width
// 					float accumulator = 0.0f;
// 					for (int c_in = 0; c_in < C_in; ++c_in)
// 					{ // Input Channel
// 						for (int kh = 0; kh < KH; ++kh)
// 						{ // Kernel Height
// 							for (int kw = 0; kw < KW; ++kw)
// 							{ // Kernel Width
// 								accumulator += input_maybe_padded(n, c_in, h_out + kh, w_out + kw) * kernel(c_out, c_in, kh, kw);
// 							}
// 						}
// 					}
// 					output(n, c_out, h_out, w_out) = accumulator;
// 				}
// 			}
// 		}
// 	}
// 	return output;
// }

// Tensor4D_NCHW convolve_fused_1x1_3x3_with_if(const Tensor4D_NCHW &input,
// 											 const Tensor4D_NCHW &kernel_1x1,
// 											 const Tensor4D_NCHW &kernel_3x3)
// {
// 	const int N = input.getB();
// 	const int C_in = input.getC();
// 	const int H_in = input.getH();
// 	const int W_in = input.getW();
// 	const int C_out_1x1 = kernel_1x1.getB();
// 	const int C_out_3x3 = kernel_3x3.getB();
// 	const int C_out_total = C_out_1x1 + C_out_3x3;

// 	Tensor4D_NCHW padded_input = input.pad_same_3x3();
// 	Tensor4D_NCHW output(N, C_out_total, H_in, W_in);
// #pragma omp parallel for collapse(2)
// 	for (int n = 0; n < N; ++n)
// 	{
// 		for (int h_out = 0; h_out < H_in; ++h_out)
// 		{
// 			for (int w_out = 0; w_out < W_in; ++w_out)
// 			{
// 				for (int c_in = 0; c_in < C_in; ++c_in)
// 				{
// 					for (int kh = 0; kh < 3; ++kh)
// 					{
// 						for (int kw = 0; kw < 3; ++kw)
// 						{
// 							float input_val = padded_input(n, c_in, h_out + kh, w_out + kw);
// 							for (int c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
// 							{
// 								output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val * kernel_3x3(c_out3, c_in, kh, kw);
// 							}

// 							// 1x1 part (conditional)
// 							if (kh == 1 && kw == 1)
// 							{
// 								for (int c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
// 								{
// 									output(n, c_out1, h_out, w_out) += input_val * kernel_1x1(c_out1, c_in, 0, 0);
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return output;
// }

// Tensor4D_NCHW convolve_fused_1x1_3x3_no_if(const Tensor4D_NCHW &input,
// 										   const Tensor4D_NCHW &kernel_1x1,
// 										   const Tensor4D_NCHW &kernel_3x3)
// {
// 	const int N = input.getB();
// 	const int C_in = input.getC();
// 	const int H_in = input.getH();
// 	const int W_in = input.getW();
// 	const int C_out_1x1 = kernel_1x1.getB();
// 	const int C_out_3x3 = kernel_3x3.getB();
// 	const int C_out_total = C_out_1x1 + C_out_3x3;

// 	Tensor4D_NCHW padded_input = input.pad_same_3x3();
// 	Tensor4D_NCHW output(N, C_out_total, H_in, W_in);
// #pragma omp parallel for collapse(2)
// 	for (int n = 0; n < N; ++n)
// 	{
// 		for (int h_out = 0; h_out < H_in; ++h_out)
// 		{
// 			for (int w_out = 0; w_out < W_in; ++w_out)
// 			{
// 				for (int c_in = 0; c_in < C_in; ++c_in)
// 				{
// 					// 3x3 part
// 					for (int kh = 0; kh < 3; ++kh)
// 					{
// 						for (int kw = 0; kw < 3; ++kw)
// 						{
// 							float input_val_3x3 = padded_input(n, c_in, h_out + kh, w_out + kw);
// 							for (int c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
// 							{
// 								output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val_3x3 * kernel_3x3(c_out3, c_in, kh, kw);
// 							}
// 						}
// 					}

// 					// 1x1 part
// 					float input_val_center = padded_input(n, c_in, h_out + 1, w_out + 1);
// 					for (int c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
// 					{
// 						output(n, c_out1, h_out, w_out) += input_val_center * kernel_1x1(c_out1, c_in, 0, 0);
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return output;
// }

static std::vector<float> pad_nhwc(const std::vector<float> &input,
								   int N, int H, int W, int C, int pad)
{
	const int H_p = H + 2 * pad;
	const int W_p = W + 2 * pad;
	std::vector<float> out(N * H_p * W_p * C, 0.0f);

	for (int n = 0; n < N; ++n)
	{
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				for (int c = 0; c < C; ++c)
				{
					out[n * (H_p * W_p * C) + (h + pad) * (W_p * C) + (w + pad) * C + c] = input[n * (H * W * C) + h * (W * C) + w * C + c];
				}
			}
		}
	}
	return out;
}
// padded_input  : N x (H_in+2) x (W_in+2) x C_in   ← уже заподджен снаружи
// kernel_1x1    : C_in x C_out_1x1                  (HWIO)
// kernel_3x3    : 3 x 3 x C_in x C_out_3x3          (HWIO)
// output        : N x H_in x W_in x (C_out_1x1 + C_out_3x3)

void convolve_fused_1x1_3x3_param(
	const ConvParams &p,
	const float *padded_input,
	const float *kernel_1x1_HWIO,
	const float *kernel_3x3_HWIO,
	float *output,
	int C_out_1x1,
	int C_out_3x3)
{
	const int H_out = p.H_out(); // = H_original
	const int W_out = p.W_out(); // = W_original
	const int C_out_total = C_out_1x1 + C_out_3x3;

	// padded input strides (p.H_in/W_in — уже с паддингом)
	const int in_stride_n = p.H_in * p.W_in * p.C_in;
	const int in_stride_h = p.W_in * p.C_in;
	const int in_stride_w = p.C_in;

	// 3×3 kernel HWIO strides
	const int k3_stride_kh = p.KH * p.C_in * C_out_3x3;
	const int k3_stride_kw = p.C_in * C_out_3x3;
	const int k3_stride_ci = C_out_3x3;

	// 1×1 kernel HWIO stride
	const int k1_stride_ci = C_out_1x1;

	// output strides
	const int out_stride_n = H_out * W_out * C_out_total;
	const int out_stride_h = W_out * C_out_total;
	const int out_stride_w = C_out_total;

#pragma omp parallel for collapse(3)
	for (int n = 0; n < p.N; ++n)
		for (int h_out = 0; h_out < H_out; ++h_out)
			for (int w_out = 0; w_out < W_out; ++w_out)
			{
				std::vector<float> acc(C_out_total, 0.0f);

				const float *in_n = padded_input + n * in_stride_n;

				// ── 3×3 ───────────────────────────────────────────────────────────
				for (int kh = 0; kh < p.KH; ++kh)
				{
					const float *in_h = in_n + (h_out + kh) * in_stride_h;
					const float *k3_kh = kernel_3x3_HWIO + kh * k3_stride_kh;

					for (int kw = 0; kw < p.KW; ++kw)
					{
						const float *in_w = in_h + (w_out + kw) * in_stride_w;
						const float *k3_kw = k3_kh + kw * k3_stride_kw;

						for (int c_in = 0; c_in < p.C_in; ++c_in)
						{
							const float v = in_w[c_in];
							const float *k3 = k3_kw + c_in * k3_stride_ci;
							for (int c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
								acc[C_out_1x1 + c_out3] += v * k3[c_out3];
						}
					}
				}

				// ── 1×1  (центр паддженного патча == оригинальный пиксель) ────────
				const float *in_center = in_n + (h_out + 1) * in_stride_h + (w_out + 1) * in_stride_w;

				for (int c_in = 0; c_in < p.C_in; ++c_in)
				{
					const float v = in_center[c_in];
					const float *k1 = kernel_1x1_HWIO + c_in * k1_stride_ci;
					for (int c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
						acc[c_out1] += v * k1[c_out1];
				}

				// ── запись ────────────────────────────────────────────────────────
				float *out_ptr = output + n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w;

				for (int c = 0; c < C_out_total; ++c)
					out_ptr[c] = acc[c];
			}
}

// ─── vector version ───────────────────────────────────────────────────────────
//
// padded_input_NHWC : N x (H_in+2) x (W_in+2) x C_in   ← уже заподджен снаружи
// kernel_1x1_HWIO   : C_in x C_out_1x1
// kernel_3x3_HWIO   : 3 x 3 x C_in x C_out_3x3
// output            : заполняется внутри, размер N x H_in x W_in x (C_out_1x1+C_out_3x3)

std::vector<float> convolve_fused_1x1_3x3(
	const std::vector<float> &padded_input_NHWC,
	const std::vector<float> &kernel_1x1_HWIO,
	const std::vector<float> &kernel_3x3_HWIO,
	std::vector<float> &output,
	int N, int H_in, int W_in, int C_in,
	int C_out_1x1, int C_out_3x3)
{
	const int C_out_total = C_out_1x1 + C_out_3x3;
	const int H_pad = H_in + 2;
	const int W_pad = W_in + 2;

	const int in_stride_n = H_pad * W_pad * C_in;
	const int in_stride_h = W_pad * C_in;
	const int in_stride_w = C_in;

	const int k3_stride_kh = 3 * C_in * C_out_3x3;
	const int k3_stride_kw = C_in * C_out_3x3;
	const int k3_stride_ci = C_out_3x3;

	const int k1_stride_ci = C_out_1x1;

	const int out_stride_n = H_in * W_in * C_out_total;
	const int out_stride_h = W_in * C_out_total;
	const int out_stride_w = C_out_total;

	output.assign(N * H_in * W_in * C_out_total, 0.0f);

#pragma omp parallel for collapse(3)
	for (int n = 0; n < N; ++n)
	{
		for (int h_out = 0; h_out < H_in; ++h_out)
		{
			for (int w_out = 0; w_out < W_in; ++w_out)
			{
				std::vector<float> acc(C_out_total, 0.0f);

				const float *in_n = padded_input_NHWC.data() + n * in_stride_n;

				// ── 3×3 ───────────────────────────────────────────────────────────
				for (int kh = 0; kh < 3; ++kh)
				{
					const float *in_h = in_n + (h_out + kh) * in_stride_h;
					const float *k3_kh = kernel_3x3_HWIO.data() + kh * k3_stride_kh;

					for (int kw = 0; kw < 3; ++kw)
					{
						const float *in_w = in_h + (w_out + kw) * in_stride_w;
						const float *k3_kw = k3_kh + kw * k3_stride_kw;

						for (int c_in = 0; c_in < C_in; ++c_in)
						{
							const float v = in_w[c_in];
							const float *k3 = k3_kw + c_in * k3_stride_ci;
							for (int c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
							{
								acc[C_out_1x1 + c_out3] += v * k3[c_out3];
							}
						}
					}
				}

				// ── 1×1 ───────────────────────────────────────────────────────────
				const float *in_center = in_n + (h_out + 1) * in_stride_h + (w_out + 1) * in_stride_w;

				for (int c_in = 0; c_in < C_in; ++c_in)
				{
					const float v = in_center[c_in];
					const float *k1 = kernel_1x1_HWIO.data() + c_in * k1_stride_ci;
					for (int c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
					{
						acc[c_out1] += v * k1[c_out1];
					}
				}

				// ── запись ────────────────────────────────────────────────────────
				float *out_ptr = output.data() + n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w;

				for (int c = 0; c < C_out_total; ++c)
				{
					out_ptr[c] = acc[c];
				}
			}
		}
	}

	return output;
}

void convolve_basic_param(const ConvParams &p, const float *input, const float *kernel,
						  float *output, int C_out_curr)
{
	const int H_out = p.H_out();
	const int W_out = p.W_out();

	const int in_stride_n = p.H_in * p.W_in * p.C_in;
	const int in_stride_h = p.W_in * p.C_in;
	const int in_stride_w = p.C_in;

	const int k_stride_kh = p.KW * p.C_in * C_out_curr;
	const int k_stride_kw = p.C_in * C_out_curr;
	const int k_stride_ci = C_out_curr;

	const int out_stride_n = H_out * W_out * C_out_curr;
	const int out_stride_h = W_out * C_out_curr;
	const int out_stride_w = C_out_curr;

#pragma omp parallel for collapse(3)
	for (int n = 0; n < p.N; n++)
	{
		for (int h_out = 0; h_out < H_out; h_out++)
		{
			for (int w_out = 0; w_out < W_out; w_out++)
			{
				std::vector<float> accumulator(C_out_curr, 0.0f);

				const float *in_base_n = input + n * in_stride_n;
				for (int kh = 0; kh < p.KH; kh++)
				{
					const int h_in = h_out + kh;
					const float *in_base_h = in_base_n + h_in * in_stride_h;
					const float *k_base_kh = kernel + kh * k_stride_kh;

					for (int kw = 0; kw < p.KW; kw++)
					{
						const int w_in = w_out + kw;
						const float *in_base_w = in_base_h + w_in * in_stride_w;
						const float *k_base_kw = k_base_kh + kw * k_stride_kw;

						for (int c_in = 0; c_in < p.C_in; c_in++)
						{
							const float input_val = in_base_w[c_in];
							const float *k_base_ci = k_base_kw + c_in * k_stride_ci;
							for (int c_out = 0; c_out < C_out_curr; c_out++)
							{
								accumulator[c_out] += input_val * k_base_ci[c_out];
							}
						}
					}
				}

				float *out_ptr = output + n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w;
				for (int c_out = 0; c_out < C_out_curr; c_out++)
				{
					out_ptr[c_out] = accumulator[c_out];
				}
			}
		}
	}
}

std::vector<float> convolve_basic(const std::vector<float> &input_NHWC,
								  const std::vector<float> &kernel_HWIO,
								  std::vector<float> &output,
								  int C_out_curr)
{
	const int in_stride_n = H_in * W_in * C_in;
	const int in_stride_h = W_in * C_in;
	const int in_stride_w = C_in;

	const int k_stride_kh = KW * C_in * C_out_curr;
	const int k_stride_kw = C_in * C_out_curr;
	const int k_stride_ci = C_out_curr;

	const int out_stride_n = H_out * W_out * C_out_curr;
	const int out_stride_h = W_out * C_out_curr;
	const int out_stride_w = C_out_curr;

	output.assign(N * H_out * W_out * C_out_curr, 0.0f);

#pragma omp parallel for collapse(3)
	for (int n = 0; n < N; n++)
	{
		for (int h_out = 0; h_out < H_out; h_out++)
		{
			for (int w_out = 0; w_out < W_out; w_out++)
			{
				std::vector<float> accumulator(C_out_curr, 0.0f);

				const float *in_base_n = input_NHWC.data() + n * in_stride_n;
				for (int kh = 0; kh < KH; kh++)
				{
					const int h_in = h_out + kh;
					const float *in_base_h = in_base_n + h_in * in_stride_h;
					const float *k_base_kh = kernel_HWIO.data() + kh * k_stride_kh;

					for (int kw = 0; kw < KW; kw++)
					{
						const int w_in = w_out + kw;
						const float *in_base_w = in_base_h + w_in * in_stride_w;
						const float *k_base_kw = k_base_kh + kw * k_stride_kw;

						for (int c_in = 0; c_in < C_in; c_in++)
						{
							const float input_val = in_base_w[c_in];
							const float *k_base_ci = k_base_kw + c_in * k_stride_ci;
							for (int c_out = 0; c_out < C_out_curr; c_out++)
							{
								accumulator[c_out] += input_val * k_base_ci[c_out];
							}
						}
					}
				}

				float *out_ptr = output.data() + n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w;
				for (int c_out = 0; c_out < C_out_curr; c_out++)
				{
					out_ptr[c_out] = accumulator[c_out];
				}
			}
		}
	}
	return output;
}

std::vector<float> conv_optimized_w_c(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr)
{
	int C_out_block_curr = C_out_curr / BLOCK_SIZE;
#pragma omp parallel for collapse(3)
	for (int n = 0; n < N; n++)
	{
		for (int c_out_block = 0; c_out_block < C_out_block_curr; c_out_block++)
		{
			for (int h_out = 0; h_out < H_out; h_out++)
			{
				std::array<float, W_out * BLOCK_SIZE> accum_block = {0};
				for (int c_in_block = 0; c_in_block < C_in_block; c_in_block++)
				{
					for (int kh = 0; kh < KH; kh++)
					{
						int h_in_coord = h_out + kh;
						for (int kw = 0; kw < KW; kw++)
						{
							for (int w_out = 0; w_out < W_out; w_out++)
							{
								int w_in_coord = w_out + kw;
								for (int c_in_inner = 0; c_in_inner < BLOCK_SIZE; c_in_inner++)
								{
									float input_val = input_NCHWc[n * C_in_block * H_out * BLOCK_SIZE * W_out + c_in_block * H_in * BLOCK_SIZE * W_in + h_in_coord * W_in * BLOCK_SIZE + w_in_coord * BLOCK_SIZE + c_in_inner];
#pragma omp simd simdlen(8)
									for (int c_out_inner = 0; c_out_inner < BLOCK_SIZE; c_out_inner++)
									{
										// для каждого w_out значение вычислятеся заново
										float kernel_val = kernel_OIHWio[c_out_block * C_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + c_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + kh * KW * BLOCK_SIZE * BLOCK_SIZE + kw * BLOCK_SIZE * BLOCK_SIZE + c_in_inner * BLOCK_SIZE + c_out_inner];
										accum_block[w_out * BLOCK_SIZE + c_out_inner] += input_val * kernel_val;
									}
								}
							}
						}
					}
				}
				for (int w_out = 0; w_out < W_out; w_out++)
				{
#pragma omp simd simdlen(8)
					for (int c_out_inner = 0; c_out_inner < BLOCK_SIZE; c_out_inner++)
					{
						output[n * H_out * W_out * C_out_curr + h_out * W_out * C_out_curr + w_out * C_out_curr + c_out_block * BLOCK_SIZE + c_out_inner] = accum_block[w_out * BLOCK_SIZE + c_out_inner];
					}
				}
			}
		}
	}
	return output;
}

std::vector<float> conv_optimized_c_w(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr)
{
	int C_out_block_curr = C_out_curr / BLOCK_SIZE;
#pragma omp parallel for collapse(3)
	for (int n = 0; n < N; n++)
	{
		for (int c_out_block = 0; c_out_block < C_out_block_curr; c_out_block++)
		{
			for (int h_out = 0; h_out < H_out; h_out++)
			{
				std::array<float, W_out * BLOCK_SIZE> accum_block = {0};
				for (int c_in_block = 0; c_in_block < C_in_block; c_in_block++)
				{
					for (int kh = 0; kh < KH; kh++)
					{
						int h_in_coord = h_out + kh;
						for (int kw = 0; kw < KW; kw++)
						{
							for (int c_in_inner = 0; c_in_inner < BLOCK_SIZE; c_in_inner++)
							{
								const float *tmp_ptr = &kernel_OIHWio[c_out_block * C_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + c_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + kh * KW * BLOCK_SIZE * BLOCK_SIZE + kw * BLOCK_SIZE * BLOCK_SIZE + c_in_inner * BLOCK_SIZE];
								for (int w_out = 0; w_out < W_out; w_out++)
								{
									int w_in_coord = w_out + kw;
									float input_val = input_NCHWc[n * C_in_block * H_out * BLOCK_SIZE * W_out + c_in_block * H_in * BLOCK_SIZE * W_in + h_in_coord * W_in * BLOCK_SIZE + w_in_coord * BLOCK_SIZE + c_in_inner];
#pragma omp simd simdlen(8)
									for (int c_out_inner = 0; c_out_inner < BLOCK_SIZE; c_out_inner++)
									{
										float kernel_val = tmp_ptr[c_out_inner];
										// float kernel_val = kernel_OIHWio[c_out_block * C_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + c_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE + kh * KW * BLOCK_SIZE * BLOCK_SIZE + kw * BLOCK_SIZE * BLOCK_SIZE + c_in_inner * BLOCK_SIZE + c_out_inner];
										accum_block[w_out * BLOCK_SIZE + c_out_inner] += input_val * kernel_val;
									}
								}
							}
						}
					}
				}
				for (int w_out = 0; w_out < W_out; w_out++)
				{
#pragma omp simd simdlen(8)
					for (int c_out_inner = 0; c_out_inner < BLOCK_SIZE; c_out_inner++)
					{
						output[n * H_out * W_out * C_out_curr + h_out * W_out * C_out_curr + w_out * C_out_curr + c_out_block * BLOCK_SIZE + c_out_inner] = accum_block[w_out * BLOCK_SIZE + c_out_inner];
					}
				}
			}
		}
	}
	return output;
}

std::vector<float> conv_optimized_v3(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr)
{
	int C_out_block_curr = C_out_curr / BLOCK_SIZE;

	const int in_stride_n = C_in_block * H_in * W_in * BLOCK_SIZE;
	const int in_stride_c = H_in * W_in * BLOCK_SIZE;
	const int in_stride_h = W_in * BLOCK_SIZE;
	const int in_stride_w = BLOCK_SIZE;

	const int k_stride_cout = C_in_block * KH * KW * BLOCK_SIZE * BLOCK_SIZE;
	const int k_stride_cin = KH * KW * BLOCK_SIZE * BLOCK_SIZE;
	const int k_stride_kh = KW * BLOCK_SIZE * BLOCK_SIZE;
	const int k_stride_kw = BLOCK_SIZE * BLOCK_SIZE;
	const int k_stride_cinner = BLOCK_SIZE;

	const int out_stride_n = H_out * W_out * C_out_curr;
	const int out_stride_h = W_out * C_out_curr;
	const int out_stride_w = C_out_curr;
	const int out_stride_c = BLOCK_SIZE;

#pragma omp parallel for collapse(3)
	for (int n = 0; n < N; n++)
	{
		for (int c_out_block = 0; c_out_block < C_out_block_curr; c_out_block++)
		{
			for (int h_out = 0; h_out < H_out; h_out++)
			{
				alignas(32) float accum_block[W_out * BLOCK_SIZE];
				for (int i = 0; i < W_out * BLOCK_SIZE; i++)
					accum_block[i] = 0.0f;

				for (int c_in_block = 0; c_in_block < C_in_block; c_in_block++)
				{
					const float *in_base_c = &input_NCHWc[n * in_stride_n + c_in_block * in_stride_c];
					const float *k_base_c = &kernel_OIHWio[c_out_block * k_stride_cout + c_in_block * k_stride_cin];

					for (int kh = 0; kh < KH; kh++)
					{
						int h_in_coord = h_out + kh;
						const float *in_base_h = in_base_c + h_in_coord * in_stride_h;
						const float *k_base_kh = k_base_c + kh * k_stride_kh;

						for (int kw = 0; kw < KW; kw++)
						{
							const float *k_base_kw = k_base_kh + kw * k_stride_kw;

							int w_out_blk = 0;
							for (; w_out_blk <= W_out - 4; w_out_blk += 4)
							{
								__m256 acc0 = _mm256_load_ps(&accum_block[(w_out_blk + 0) * BLOCK_SIZE]);
								__m256 acc1 = _mm256_load_ps(&accum_block[(w_out_blk + 1) * BLOCK_SIZE]);
								__m256 acc2 = _mm256_load_ps(&accum_block[(w_out_blk + 2) * BLOCK_SIZE]);
								__m256 acc3 = _mm256_load_ps(&accum_block[(w_out_blk + 3) * BLOCK_SIZE]);

								const float *in_ptr0 = in_base_h + (w_out_blk + 0 + kw) * in_stride_w;
								const float *in_ptr1 = in_base_h + (w_out_blk + 1 + kw) * in_stride_w;
								const float *in_ptr2 = in_base_h + (w_out_blk + 2 + kw) * in_stride_w;
								const float *in_ptr3 = in_base_h + (w_out_blk + 3 + kw) * in_stride_w;

								for (int c_in_inner = 0; c_in_inner < BLOCK_SIZE; c_in_inner++)
								{
									__m256 k_vec = _mm256_loadu_ps(k_base_kw + c_in_inner * k_stride_cinner);

									__m256 in0 = _mm256_set1_ps(in_ptr0[c_in_inner]);
									acc0 = _mm256_fmadd_ps(in0, k_vec, acc0);

									__m256 in1 = _mm256_set1_ps(in_ptr1[c_in_inner]);
									acc1 = _mm256_fmadd_ps(in1, k_vec, acc1);

									__m256 in2 = _mm256_set1_ps(in_ptr2[c_in_inner]);
									acc2 = _mm256_fmadd_ps(in2, k_vec, acc2);

									__m256 in3 = _mm256_set1_ps(in_ptr3[c_in_inner]);
									acc3 = _mm256_fmadd_ps(in3, k_vec, acc3);
								}

								_mm256_store_ps(&accum_block[(w_out_blk + 0) * BLOCK_SIZE], acc0);
								_mm256_store_ps(&accum_block[(w_out_blk + 1) * BLOCK_SIZE], acc1);
								_mm256_store_ps(&accum_block[(w_out_blk + 2) * BLOCK_SIZE], acc2);
								_mm256_store_ps(&accum_block[(w_out_blk + 3) * BLOCK_SIZE], acc3);
							}

							for (; w_out_blk < W_out; w_out_blk++)
							{
								__m256 acc = _mm256_load_ps(&accum_block[w_out_blk * BLOCK_SIZE]);
								const float *in_ptr = in_base_h + (w_out_blk + kw) * in_stride_w;

								for (int c_in_inner = 0; c_in_inner < BLOCK_SIZE; c_in_inner++)
								{
									__m256 k_vec = _mm256_loadu_ps(k_base_kw + c_in_inner * k_stride_cinner);
									__m256 in_vec = _mm256_set1_ps(in_ptr[c_in_inner]);
									acc = _mm256_fmadd_ps(in_vec, k_vec, acc);
								}
								_mm256_store_ps(&accum_block[w_out_blk * BLOCK_SIZE], acc);
							}
						}
					}
				}

				for (int w_out = 0; w_out < W_out; w_out++)
				{
					float *out_ptr = &output[n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w + c_out_block * out_stride_c];
					float *acc_ptr = &accum_block[w_out * BLOCK_SIZE];
					_mm256_storeu_ps(out_ptr, _mm256_load_ps(acc_ptr));
				}
			}
		}
	}
	return output;
}
// ================================================================
// Parameterized versions (accept ConvParams instead of globals)
// ================================================================

void conv_param_w_c(const ConvParams &p, const float *input, const float *kernel,
					float *output, int C_out_curr)
{
	const int BS = BLOCK_SIZE;
	const int C_out_block_curr = C_out_curr / BS;
	const int c_in_blk = p.C_in_block();
	const int h_out = p.H_out();
	const int w_out = p.W_out();

#pragma omp parallel for collapse(3)
	for (int n = 0; n < p.N; n++)
	{
		for (int cob = 0; cob < C_out_block_curr; cob++)
		{
			for (int ho = 0; ho < h_out; ho++)
			{
				std::vector<float> accum_block(w_out * BS, 0.0f);

				for (int cib = 0; cib < c_in_blk; cib++)
				{
					for (int kh = 0; kh < p.KH; kh++)
					{
						int h_in_coord = ho + kh;
						for (int kw = 0; kw < p.KW; kw++)
						{
							for (int wo = 0; wo < w_out; wo++)
							{
								int w_in_coord = wo + kw;
								for (int ci = 0; ci < BS; ci++)
								{
									float iv = input[n * c_in_blk * p.H_in * p.W_in * BS + cib * p.H_in * p.W_in * BS + h_in_coord * p.W_in * BS + w_in_coord * BS + ci];
#pragma omp simd simdlen(8)
									for (int co = 0; co < BS; co++)
									{
										float kv = kernel[cob * c_in_blk * p.KH * p.KW * BS * BS + cib * p.KH * p.KW * BS * BS + kh * p.KW * BS * BS + kw * BS * BS + ci * BS + co];
										accum_block[wo * BS + co] += iv * kv;
									}
								}
							}
						}
					}
				}
				for (int wo = 0; wo < w_out; wo++)
				{
#pragma omp simd simdlen(8)
					for (int co = 0; co < BS; co++)
					{
						output[n * h_out * w_out * C_out_curr + ho * w_out * C_out_curr + wo * C_out_curr + cob * BS + co] = accum_block[wo * BS + co];
					}
				}
			}
		}
	}
}

void conv_param_c_w(const ConvParams &p, const float *input, const float *kernel,
					float *output, int C_out_curr)
{
	const int BS = BLOCK_SIZE;
	const int C_out_block_curr = C_out_curr / BS;
	const int c_in_blk = p.C_in_block();
	const int h_out = p.H_out();
	const int w_out = p.W_out();

#pragma omp parallel for collapse(3)
	for (int n = 0; n < p.N; n++)
	{
		for (int cob = 0; cob < C_out_block_curr; cob++)
		{
			for (int ho = 0; ho < h_out; ho++)
			{
				std::vector<float> accum_block(w_out * BS, 0.0f);

				for (int cib = 0; cib < c_in_blk; cib++)
				{
					for (int kh = 0; kh < p.KH; kh++)
					{
						int h_in_coord = ho + kh;
						for (int kw = 0; kw < p.KW; kw++)
						{
							for (int ci = 0; ci < BS; ci++)
							{
								for (int wo = 0; wo < w_out; wo++)
								{
									int w_in_coord = wo + kw;
									float iv = input[n * c_in_blk * p.H_in * p.W_in * BS + cib * p.H_in * p.W_in * BS + h_in_coord * p.W_in * BS + w_in_coord * BS + ci];
#pragma omp simd simdlen(8)
									for (int co = 0; co < BS; co++)
									{
										float kv = kernel[cob * c_in_blk * p.KH * p.KW * BS * BS + cib * p.KH * p.KW * BS * BS + kh * p.KW * BS * BS + kw * BS * BS + ci * BS + co];
										accum_block[wo * BS + co] += iv * kv;
									}
								}
							}
						}
					}
				}
				for (int wo = 0; wo < w_out; wo++)
				{
#pragma omp simd simdlen(8)
					for (int co = 0; co < BS; co++)
					{
						output[n * h_out * w_out * C_out_curr + ho * w_out * C_out_curr + wo * C_out_curr + cob * BS + co] = accum_block[wo * BS + co];
					}
				}
			}
		}
	}
}

void conv_param_v3(const ConvParams &p, const float *input, const float *kernel,
				   float *output, int C_out_curr)
{
	const int BS = BLOCK_SIZE;
	const int C_out_block_curr = C_out_curr / BS;
	const int c_in_blk = p.C_in_block();
	const int h_out = p.H_out();
	const int w_out = p.W_out();

	const int in_stride_n = c_in_blk * p.H_in * p.W_in * BS;
	const int in_stride_c = p.H_in * p.W_in * BS;
	const int in_stride_h = p.W_in * BS;
	const int in_stride_w = BS;

	const int k_stride_cout = c_in_blk * p.KH * p.KW * BS * BS;
	const int k_stride_cin = p.KH * p.KW * BS * BS;
	const int k_stride_kh = p.KW * BS * BS;
	const int k_stride_kw = BS * BS;
	const int k_stride_cinner = BS;

	const int out_stride_n = h_out * w_out * C_out_curr;
	const int out_stride_h = w_out * C_out_curr;
	const int out_stride_w = C_out_curr;
	const int out_stride_c = BS;

#pragma omp parallel for collapse(3)
	for (int n = 0; n < p.N; n++)
	{
		for (int cob = 0; cob < C_out_block_curr; cob++)
		{
			for (int ho = 0; ho < h_out; ho++)
			{
				std::vector<float> accum_block(w_out * BS, 0.0f);

				for (int cib = 0; cib < c_in_blk; cib++)
				{
					const float *in_base_c = &input[n * in_stride_n + cib * in_stride_c];
					const float *k_base_c = &kernel[cob * k_stride_cout + cib * k_stride_cin];

					for (int kh = 0; kh < p.KH; kh++)
					{
						int h_in_coord = ho + kh;
						const float *in_base_h = in_base_c + h_in_coord * in_stride_h;
						const float *k_base_kh = k_base_c + kh * k_stride_kh;

						for (int kw = 0; kw < p.KW; kw++)
						{
							const float *k_base_kw = k_base_kh + kw * k_stride_kw;

							int wb = 0;
							for (; wb <= w_out - 4; wb += 4)
							{
								__m256 acc0 = _mm256_loadu_ps(&accum_block[(wb + 0) * BS]);
								__m256 acc1 = _mm256_loadu_ps(&accum_block[(wb + 1) * BS]);
								__m256 acc2 = _mm256_loadu_ps(&accum_block[(wb + 2) * BS]);
								__m256 acc3 = _mm256_loadu_ps(&accum_block[(wb + 3) * BS]);

								const float *ip0 = in_base_h + (wb + 0 + kw) * in_stride_w;
								const float *ip1 = in_base_h + (wb + 1 + kw) * in_stride_w;
								const float *ip2 = in_base_h + (wb + 2 + kw) * in_stride_w;
								const float *ip3 = in_base_h + (wb + 3 + kw) * in_stride_w;

								for (int ci = 0; ci < BS; ci++)
								{
									__m256 kv = _mm256_loadu_ps(k_base_kw + ci * k_stride_cinner);
									acc0 = _mm256_fmadd_ps(_mm256_set1_ps(ip0[ci]), kv, acc0);
									acc1 = _mm256_fmadd_ps(_mm256_set1_ps(ip1[ci]), kv, acc1);
									acc2 = _mm256_fmadd_ps(_mm256_set1_ps(ip2[ci]), kv, acc2);
									acc3 = _mm256_fmadd_ps(_mm256_set1_ps(ip3[ci]), kv, acc3);
								}

								_mm256_storeu_ps(&accum_block[(wb + 0) * BS], acc0);
								_mm256_storeu_ps(&accum_block[(wb + 1) * BS], acc1);
								_mm256_storeu_ps(&accum_block[(wb + 2) * BS], acc2);
								_mm256_storeu_ps(&accum_block[(wb + 3) * BS], acc3);
							}
							for (; wb < w_out; wb++)
							{
								__m256 acc = _mm256_loadu_ps(&accum_block[wb * BS]);
								const float *ip = in_base_h + (wb + kw) * in_stride_w;
								for (int ci = 0; ci < BS; ci++)
								{
									__m256 kv = _mm256_loadu_ps(k_base_kw + ci * k_stride_cinner);
									acc = _mm256_fmadd_ps(_mm256_set1_ps(ip[ci]), kv, acc);
								}
								_mm256_storeu_ps(&accum_block[wb * BS], acc);
							}
						}
					}
				}

				for (int wo = 0; wo < w_out; wo++)
				{
					float *out_ptr = &output[n * out_stride_n + ho * out_stride_h + wo * out_stride_w + cob * out_stride_c];
					_mm256_storeu_ps(out_ptr, _mm256_loadu_ps(&accum_block[wo * BS]));
				}
			}
		}
	}
}
