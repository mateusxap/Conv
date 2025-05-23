#include "Conv.hpp"

// ------ Tensor4D impl ------

double Tensor4D::check_difference(const Tensor4D &t2)
{
	if (this->size() != t2.size() || this->size() == 0)
	{
		return -1.0;
	}
	double diff_sum = 0.0;
	for (size_t i = 0; i < this->size(); ++i)
	{
		diff_sum += std::fabs(this->data[i] - t2.data[i]);
	}
	return diff_sum / this->size();
}

// ------ Tensor4D_NCHW impl ------

inline float &Tensor4D_NCHW::operator()(size_t n, size_t c, size_t h, size_t w)
{
	return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
}

inline const float &Tensor4D_NCHW::operator()(size_t n, size_t c, size_t h, size_t w) const
{
	return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
}

Tensor4D_NCHW Tensor4D_NCHW::pad_same_3x3() const
{
	Tensor4D_NCHW padded_output(this->B_dim, this->C_dim, this->H_dim + 2, this->W_dim + 2);
	for (size_t n = 0; n < this->B_dim; ++n)
	{
		for (size_t c = 0; c < this->C_dim; ++c)
		{
			for (size_t h = 0; h < this->H_dim; ++h)
			{
				for (size_t w = 0; w < this->W_dim; ++w)
				{
					padded_output(n, c, h + 1, w + 1) = (*this)(n, c, h, w);
				}
			}
		}
	}
	return padded_output;
}

Tensor4D_NCHW Tensor4D_NCHW::concatenateChannels(const Tensor4D &t2) const
{
	assert(this->B_dim == t2.getB() && this->H_dim == t2.getH() && this->W_dim == t2.getW());
	size_t N = this->B_dim, C1 = this->C_dim, C2 = t2.getC(), H = this->H_dim, W = this->W_dim;
	size_t C_total = C1 + C2;
	Tensor4D_NCHW result(N, C_total, H, W);
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t h = 0; h < H; ++h)
		{
			for (size_t w = 0; w < W; ++w)
			{
				for (size_t c = 0; c < C1; ++c)
					result(n, c, h, w) = (*this)(n, c, h, w);
				for (size_t c = 0; c < C2; ++c)
					result(n, C1 + c, h, w) = t2(n, c, h, w);
			}
		}
	}
	return result;
}

Tensor4D_NCHW convolve_basic(const Tensor4D_NCHW &input_maybe_padded, const Tensor4D_NCHW &kernel)
{
	const size_t N = input_maybe_padded.getB();
	const size_t C_in = input_maybe_padded.getC();
	const size_t H_in = input_maybe_padded.getH();
	const size_t W_in = input_maybe_padded.getW();
	const size_t C_out = kernel.getB();
	const size_t KH = kernel.getH();
	const size_t KW = kernel.getW();
	const size_t H_out = H_in - KH + 1;
	const size_t W_out = W_in - KW + 1;

	Tensor4D_NCHW output(N, C_out, H_out, W_out);

	for (size_t n = 0; n < N; ++n)
	{ // Batch
		for (size_t c_out = 0; c_out < C_out; ++c_out)
		{ // Output Channel
			for (size_t h_out = 0; h_out < H_out; ++h_out)
			{ // Output Height
				for (size_t w_out = 0; w_out < W_out; ++w_out)
				{ // Output Width
					float accumulator = 0.0f;
					for (size_t c_in = 0; c_in < C_in; ++c_in)
					{ // Input Channel
						for (size_t kh = 0; kh < KH; ++kh)
						{ // Kernel Height
							for (size_t kw = 0; kw < KW; ++kw)
							{ // Kernel Width
								accumulator += input_maybe_padded(n, c_in, h_out + kh, w_out + kw) * kernel(c_out, c_in, kh, kw);
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

Tensor4D_NCHW convolve_fused_1x1_3x3_with_if(const Tensor4D_NCHW &input,
																						 const Tensor4D_NCHW &kernel_1x1,
																						 const Tensor4D_NCHW &kernel_3x3)
{
	const size_t N = input.getB();
	const size_t C_in = input.getC();
	const size_t H_in = input.getH();
	const size_t W_in = input.getW();
	const size_t C_out_1x1 = kernel_1x1.getB();
	const size_t C_out_3x3 = kernel_3x3.getB();
	const size_t C_out_total = C_out_1x1 + C_out_3x3;

	Tensor4D_NCHW padded_input = input.pad_same_3x3();
	Tensor4D_NCHW output(N, C_out_total, H_in, W_in);

	for (size_t n = 0; n < N; ++n)
	{
		for (size_t h_out = 0; h_out < H_in; ++h_out)
		{
			for (size_t w_out = 0; w_out < W_in; ++w_out)
			{
				for (size_t c_in = 0; c_in < C_in; ++c_in)
				{
					for (size_t kh = 0; kh < 3; ++kh)
					{
						for (size_t kw = 0; kw < 3; ++kw)
						{
							float input_val = padded_input(n, c_in, h_out + kh, w_out + kw);
							for (size_t c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
							{
								output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val * kernel_3x3(c_out3, c_in, kh, kw);
							}

							// 1x1 part (conditional)
							if (kh == 1 && kw == 1)
							{
								for (size_t c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
								{
									output(n, c_out1, h_out, w_out) += input_val * kernel_1x1(c_out1, c_in, 0, 0);
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

Tensor4D_NCHW convolve_fused_1x1_3x3_no_if(const Tensor4D_NCHW &input,
																					 const Tensor4D_NCHW &kernel_1x1,
																					 const Tensor4D_NCHW &kernel_3x3)
{
	const size_t N = input.getB();
	const size_t C_in = input.getC();
	const size_t H_in = input.getH();
	const size_t W_in = input.getW();
	const size_t C_out_1x1 = kernel_1x1.getB();
	const size_t C_out_3x3 = kernel_3x3.getB();
	const size_t C_out_total = C_out_1x1 + C_out_3x3;

	Tensor4D_NCHW padded_input = input.pad_same_3x3();
	Tensor4D_NCHW output(N, C_out_total, H_in, W_in);

	for (size_t n = 0; n < N; ++n)
	{
		for (size_t h_out = 0; h_out < H_in; ++h_out)
		{
			for (size_t w_out = 0; w_out < W_in; ++w_out)
			{
				for (size_t c_in = 0; c_in < C_in; ++c_in)
				{
					// 3x3 part
					for (size_t kh = 0; kh < 3; ++kh)
					{
						for (size_t kw = 0; kw < 3; ++kw)
						{
							float input_val_3x3 = padded_input(n, c_in, h_out + kh, w_out + kw);
							for (size_t c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
							{
								output(n, C_out_1x1 + c_out3, h_out, w_out) += input_val_3x3 * kernel_3x3(c_out3, c_in, kh, kw);
							}
						}
					}

					// 1x1 part
					float input_val_center = padded_input(n, c_in, h_out + 1, w_out + 1);
					for (size_t c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
					{
						output(n, c_out1, h_out, w_out) += input_val_center * kernel_1x1(c_out1, c_in, 0, 0);
					}
				}
			}
		}
	}
	return output;
}

// ------ Tensor4D_NHWC impl ------

inline float &Tensor4D_NHWC::operator()(size_t n, size_t h, size_t w, size_t c)
{
	return data[n * H_dim * W_dim * C_dim + h * W_dim * C_dim + w * C_dim + c];
}

inline const float &Tensor4D_NHWC::operator()(size_t n, size_t h, size_t w, size_t c) const
{
	return data[n * H_dim * W_dim * C_dim + h * W_dim * C_dim + w * C_dim + c];
}

Tensor4D_NHWC Tensor4D_NHWC::pad_same_3x3() const
{
	Tensor4D_NHWC padded_output(this->B_dim, this->H_dim + 2, this->W_dim + 2, this->C_dim);
	for (size_t n = 0; n < this->B_dim; ++n)
	{
		for (size_t h = 0; h < this->H_dim; ++h)
		{
			for (size_t w = 0; w < this->W_dim; ++w)
			{
				for (size_t c = 0; c < this->C_dim; ++c)
				{
					padded_output(n, h + 1, w + 1, c) = (*this)(n, h, w, c);
				}
			}
		}
	}
	return padded_output;
}

Tensor4D_NHWC Tensor4D_NHWC::concatenateChannels(const Tensor4D &t2) const
{
	assert(this->B_dim == t2.getB() && this->H_dim == t2.getH() && this->W_dim == t2.getW());
	size_t N = this->B_dim, C1 = this->C_dim, C2 = t2.getC(), H = this->H_dim, W = this->W_dim;
	size_t C_total = C1 + C2;
	Tensor4D_NHWC result(N, H, W, C_total);
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t h = 0; h < H; ++h)
		{
			for (size_t w = 0; w < W; ++w)
			{
				for (size_t c = 0; c < C1; ++c)
					result(n, h, w, c) = (*this)(n, h, w, c);
				for (size_t c = 0; c < C2; ++c)
					result(n, h, w, C1 + c) = t2(n, h, w, c);
			}
		}
	}
	return result;
}

Tensor4D_NHWC convolve_basic(const Tensor4D_NHWC &input, const Tensor4D_HWIO &kernel)
{
	const size_t N = input.getB();
	const size_t C_in = input.getC();
	const size_t H_in = input.getH();
	const size_t W_in = input.getW();
	const size_t C_out = kernel.getB();
	const size_t KH = kernel.getH();
	const size_t KW = kernel.getW();

	const size_t H_out = H_in - KH + 1;
	const size_t W_out = W_in - KW + 1;

	Tensor4D_NHWC output(N, H_out, W_out, C_out);
	for (size_t n = 0; n < N; n++)
	{
		for (size_t h_out = 0; h_out < H_out; h_out++)
		{
			for (size_t w_out = 0; w_out < W_out; w_out++)
			{
				std::vector<float> accumulator(C_out, 0.0f);
				for (size_t kh = 0; kh < KH; kh++)
				{
					for (size_t kw = 0; kw < KW; kw++)
					{
						for (size_t c_in = 0; c_in < C_in; c_in++)
						{
							float input_val = input(n, h_out + kh, w_out + kw, c_in);
							for (size_t c_out = 0; c_out < C_out; c_out++)
							{
								// accumulator += input[n][h_out + kh][w_out + kw][c_in] * kernel[kh][kw][c_in][c_out];
								accumulator[c_out] += input_val * kernel(kh, kw, c_in, c_out);
							}
						}
					}
				}
				for (size_t c_out = 0; c_out < C_out; c_out++)
				{
					output(n, h_out, w_out, c_out) = accumulator[c_out];
				}
			}
		}
	}
	return output;
}

// Tensor4D_NHWC convolve_fused_1x1_3x3_no_if(const Tensor4D_NHWC &input,
// 																					 const Tensor4D_NHWC &kernel_1x1,
// 																					 const Tensor4D_NHWC &kernel_3x3)
// {
// 	const size_t N = input.getB();
// 	const size_t C_in = input.getC();
// 	const size_t H_in = input.getH();
// 	const size_t W_in = input.getW();
// 	const size_t C_out_1x1 = kernel_1x1.getB();
// 	const size_t C_out_3x3 = kernel_3x3.getB();
// 	const size_t C_out_total = C_out_1x1 + C_out_3x3;

// 	Tensor4D_NHWC padded_input = input.pad_same_3x3();
// 	Tensor4D_NHWC output(N, H_in, W_in, C_out_total);

// 	for (size_t n = 0; n < N; ++n)
// 	{
// 		for (size_t h_out = 0; h_out < H_in; ++h_out)
// 		{
// 			for (size_t w_out = 0; w_out < W_in; ++w_out)
// 			{
// 				// 3x3 part
// 				for (size_t kh = 0; kh < 3; ++kh)
// 				{
// 					for (size_t kw = 0; kw < 3; ++kw)
// 					{
// 						for (size_t c_in = 0; c_in < C_in; ++c_in)
// 						{
// 							float input_val_3x3 = padded_input(n, h_out + kh, w_out + kw, c_in);
// 							for (size_t c_out3 = 0; c_out3 < C_out_3x3; ++c_out3)
// 							{
// 								size_t kernel_3x3_flat_idx = c_out3 * (3 * 3 * C_in) + kh * (3 * C_in) + kw * (C_in) + c_in;
// 								output(n, h_out, w_out, C_out_1x1 + c_out3) += input_val_3x3 * kernel_3x3.data[kernel_3x3_flat_idx];
// 								/*size_t kernel_3x3_full_idx = c_out3 * (C_in * 9) + c_in * 9 + kh * 3 + kw;
// 								output(n, h_out, w_out, C_out_1x1 + c_out3) += input_val_3x3 * kernel_3x3.data[kernel_3x3_full_idx]*/
// 								; // ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ (ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½)
// 							}
// 						}
// 					}
// 				}
// 				for (size_t c_in = 0; c_in < C_in; ++c_in)
// 				{ // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ (ï¿½ï¿½ï¿½. NCHW) ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
// 					// 1x1 part
// 					float input_val_center = padded_input(n, h_out + 1, w_out + 1, c_in);
// 					for (size_t c_out1 = 0; c_out1 < C_out_1x1; ++c_out1)
// 					{
// 						size_t kernel_1x1_full_idx = c_out1 * C_in + c_in;
// 						output(n, h_out, w_out, c_out1) += input_val_center * kernel_1x1.data[kernel_1x1_full_idx]; // ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return output;
// }
